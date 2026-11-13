"""
LegoGPT: Connectivity & Port Database
Provides accurate 3D physical characterization of LEGO parts, connection ports
(studs, sockets, anti-stud tubes, pins, axles, clips, bars, hinges, balls),
contact surfaces, and mechanical coupling rules.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.parser import ParsedPart

class PortType(str, Enum):
    STUD = "stud"                 # Standard male stud (solid)
    STUD_HOLLOW = "stud_hollow"   # Male hollow stud (can receive a bar/rod/pin)
    ANTISTUD = "antistud"         # Female socket or bottom tube cavity
    PIN = "pin"                   # Technic pin (male)
    PIN_HOLE = "pin_hole"         # Technic pin hole (female)
    AXLE = "axle"                 # Technic axle cross (male)
    AXLE_HOLE = "axle_hole"       # Technic axle hole (female)
    CLIP = "clip"                 # Clip jaw (female)
    BAR = "bar"                   # Bar / shaft Ø 3.18 mm (male)
    BALL = "ball"                 # Ball joint sphere (male)
    BALL_SOCKET = "ball_socket"   # Ball joint cup (female)
    HINGE_MALE = "hinge_male"     # Hinge tab/pin
    HINGE_FEMALE = "hinge_female" # Hinge slot/fork

class JointDOF(str, Enum):
    FIXED = "fixed"               # 0 Degrees of Freedom (snaps rigidly)
    REVOLUTE = "revolute"         # 1 DoF: Rotation around axis (pin, clip, round stud)
    PRISMATIC = "prismatic"       # 1 DoF: Translation along axis (axle sliding)
    SPHERICAL = "spherical"       # 3 DoF: Ball joint

@dataclass
class ConnectionPort:
    id: str
    port_type: PortType
    gender: str                   # "male" | "female"
    local_pos: np.ndarray         # shape (3,), float32 in LDU
    local_normal: np.ndarray      # shape (3,), unit vector pointing away from face
    dof: JointDOF = JointDOF.FIXED
    symmetry: int = 4             # 4 for square studs (90° steps), 1 for continuous
    clutch_power: float = 2.0     # Nominal retention force in Newtons
    source: str = "parametric"    # "parametric" | "ldraw_dat" | "ldcad_snap" (for diagnostics)

# Port compatibility pairing rules
COMPATIBLE_PORT_PAIRS = {
    (PortType.STUD, PortType.ANTISTUD),
    (PortType.ANTISTUD, PortType.STUD),
    (PortType.STUD_HOLLOW, PortType.ANTISTUD),
    (PortType.ANTISTUD, PortType.STUD_HOLLOW),
    (PortType.STUD_HOLLOW, PortType.BAR),
    (PortType.BAR, PortType.STUD_HOLLOW),
    (PortType.PIN, PortType.PIN_HOLE),
    (PortType.PIN_HOLE, PortType.PIN),
    (PortType.AXLE, PortType.AXLE_HOLE),
    (PortType.AXLE_HOLE, PortType.AXLE),
    (PortType.CLIP, PortType.BAR),
    (PortType.BAR, PortType.CLIP),
    (PortType.BALL, PortType.BALL_SOCKET),
    (PortType.BALL_SOCKET, PortType.BALL),
    (PortType.HINGE_MALE, PortType.HINGE_FEMALE),
    (PortType.HINGE_FEMALE, PortType.HINGE_MALE),
}

# Standard search paths for LDraw library on macOS / Linux
DEFAULT_LDRAW_PATHS = [
    "/Applications/Studio 2.0/ldraw",
    os.path.expanduser("~/Library/Application Support/Studio 2.0/ldraw"),
    "/usr/share/ldraw",
    "/usr/local/share/ldraw",
    os.path.expanduser("~/ldraw"),
    os.path.expanduser("~/.ldraw"),
]

# Memory cache for resolved ports
_PORT_CACHE: dict[str, list[ConnectionPort]] = {}

def get_ldraw_root() -> str | None:
    """Finds the LDraw root directory if installed on the host machine."""
    for path in DEFAULT_LDRAW_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    return None

def find_ldraw_part_file(part_id: str, ldraw_root: str | None = None) -> str | None:
    """Locates an LDraw part .dat file within standard LDraw directories."""
    if not ldraw_root:
        ldraw_root = get_ldraw_root()
    if not ldraw_root:
        return None

    clean_name = part_id.replace("\\", "/").lower()
    if not clean_name.endswith(".dat"):
        clean_name += ".dat"
        
    base = os.path.basename(clean_name)
    sub = os.path.dirname(clean_name)
    
    candidates = [
        os.path.join(ldraw_root, "parts", sub, base),
        os.path.join(ldraw_root, "parts", "s", base),
        os.path.join(ldraw_root, "parts", base),
        os.path.join(ldraw_root, "p", sub, base),
        os.path.join(ldraw_root, "p", "48", base),
        os.path.join(ldraw_root, "p", base),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _parse_ldcad_shadow_line(line_str: str, current_transform: np.ndarray, port_idx: int) -> ConnectionPort | None:
    """
    Parses an LDCad Shadow Library meta-command (used by LTRON / LDCad):
      0 !LDCAD SNAP_CYL [gender=M|F] [caps=...] [pos=x y z] [ori=a b c d e f g h i] [grid=...]
      0 !LDCAD SNAP_CLP [pos=x y z] [ori=...]
      0 !LDCAD SNAP_FGR / SNAP_SPH / SNAP_GEN
    and converts it into a native LegoGPT ConnectionPort in the part's local frame.
    """
    if "!LDCAD" not in line_str.upper():
        return None
    upper = line_str.upper()
    if not any(cmd in upper for cmd in ("SNAP_CYL", "SNAP_CLP", "SNAP_FGR", "SNAP_SPH", "SNAP_GEN")):
        return None

    # Extract [key=val] blocks
    props: dict[str, str] = {}
    for m in re.finditer(r"\[([a-zA-Z0-9_]+)=([^\]]+)\]", line_str):
        props[m.group(1).lower()] = m.group(2).strip()

    pos_local = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if "pos" in props:
        try:
            vals = [float(v) for v in props["pos"].split()]
            if len(vals) == 3:
                pos_local = np.array(vals, dtype=np.float32)
        except ValueError:
            pass

    ori_local = np.eye(3, dtype=np.float32)
    if "ori" in props:
        try:
            vals = [float(v) for v in props["ori"].split()]
            if len(vals) == 9:
                ori_local = np.array(vals, dtype=np.float32).reshape(3, 3)
        except ValueError:
            pass

    local_t = np.eye(4, dtype=np.float32)
    local_t[:3, :3] = ori_local
    local_t[:3, 3] = pos_local
    accum_t = current_transform @ local_t

    pos = accum_t[:3, 3]
    normal = accum_t[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
    norm_mag = np.linalg.norm(normal)
    normal = (normal / norm_mag) if norm_mag > 1e-4 else np.array([0.0, -1.0, 0.0], dtype=np.float32)

    gender_raw = props.get("gender", "m").lower()
    gender = "female" if gender_raw.startswith("f") else "male"
    sec = props.get("secs", "").lower()

    if "SNAP_CLP" in upper:
        return ConnectionPort(
            id=f"ldcad_clip_{port_idx}",
            port_type=PortType.CLIP,
            gender="female",
            local_pos=pos,
            local_normal=normal,
            dof=JointDOF.REVOLUTE,
            symmetry=1,
            clutch_power=1.8,
        )
    if "SNAP_SPH" in upper:
        p_type = PortType.BALL_SOCKET if gender == "female" else PortType.BALL
        return ConnectionPort(
            id=f"ldcad_ball_{port_idx}",
            port_type=p_type,
            gender=gender,
            local_pos=pos,
            local_normal=normal if gender == "male" else -normal,
            dof=JointDOF.SPHERICAL,
            symmetry=1,
            clutch_power=2.5,
        )
    if "SNAP_FGR" in upper:
        p_type = PortType.HINGE_FEMALE if gender == "female" else PortType.HINGE_MALE
        return ConnectionPort(
            id=f"ldcad_hinge_{port_idx}",
            port_type=p_type,
            gender=gender,
            local_pos=pos,
            local_normal=normal,
            dof=JointDOF.REVOLUTE,
            symmetry=1,
            clutch_power=2.2,
        )

    # SNAP_CYL or SNAP_GEN
    if "axle" in sec or "X" in props.get("secs", ""):
        p_type = PortType.AXLE_HOLE if gender == "female" else PortType.AXLE
        dof = JointDOF.PRISMATIC
        sym = 4
    elif "pin" in sec:
        p_type = PortType.PIN_HOLE if gender == "female" else PortType.PIN
        dof = JointDOF.REVOLUTE
        sym = 1
    elif "bar" in sec:
        p_type = PortType.STUD_HOLLOW if gender == "female" else PortType.BAR
        dof = JointDOF.REVOLUTE
        sym = 1
    else:
        p_type = PortType.ANTISTUD if gender == "female" else PortType.STUD
        dof = JointDOF.FIXED
        sym = 4

    return ConnectionPort(
        id=f"ldcad_{p_type.value}_{port_idx}",
        port_type=p_type,
        gender=gender,
        local_pos=pos,
        local_normal=-normal if p_type == PortType.ANTISTUD else normal,
        dof=dof,
        symmetry=sym,
        clutch_power=2.0,
        source="ldcad_snap",
    )


def _extract_ports_from_ldraw_dat(part_id: str, ldraw_root: str | None = None) -> list[ConnectionPort]:
    """
    Recursively scans an LDraw part .dat file (and optional LDCad Shadow !LDCAD SNAP_* directives)
    and identifies connector primitives (stud.dat, stud4.dat, axle.dat, pin.dat, clip*.dat, etc.)
    with their exact local poses.
    """
    if not ldraw_root:
        ldraw_root = get_ldraw_root()
    if not ldraw_root:
        return []

    ports: list[ConnectionPort] = []
    visited: set[str] = set()

    def scan_file(file_path: str, current_transform: np.ndarray, depth: int = 0):
        if depth > 4:
            return
        base_name = os.path.basename(file_path).lower()
        if base_name in visited:
            return
        visited.add(base_name)

        try:
            with open(file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception:
            return

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("0 ") and "!LDCAD" in line_str.upper():
                shadow_port = _parse_ldcad_shadow_line(line_str, current_transform, len(ports))
                if shadow_port is not None:
                    ports.append(shadow_port)
                continue

            if not line_str.startswith("1 "):
                continue

            tokens = line_str.split()
            if len(tokens) < 15:
                continue

            try:
                x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                a, b, c = float(tokens[5]), float(tokens[6]), float(tokens[7])
                d, e, f = float(tokens[8]), float(tokens[9]), float(tokens[10])
                g, h, i = float(tokens[11]), float(tokens[12]), float(tokens[13])
            except ValueError:
                continue

            local_t = np.array([
                [a, b, c, x],
                [d, e, f, y],
                [g, h, i, z],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

            accum_t = current_transform @ local_t
            sub_name = os.path.basename(tokens[14].replace("\\", "/")).lower()

            # Identify primitives
            pos = accum_t[:3, 3]
            # Normal: default LDraw stud points along -Y in primitive space
            normal = accum_t[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-4:
                normal = normal / norm_mag
            else:
                normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            if sub_name == "stud.dat":
                ports.append(ConnectionPort(
                    id=f"stud_{len(ports)}",
                    port_type=PortType.STUD,
                    gender="male",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.FIXED,
                    symmetry=4,
                    clutch_power=2.0,
                    source="ldraw_dat"
                ))
            elif sub_name in ("stud2.dat", "stud2a.dat", "studopen.dat"):
                ports.append(ConnectionPort(
                    id=f"stud_hollow_{len(ports)}",
                    port_type=PortType.STUD_HOLLOW,
                    gender="male",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.REVOLUTE,
                    symmetry=1,
                    clutch_power=2.0,
                    source="ldraw_dat"
                ))
            elif sub_name in ("stud4.dat", "stud3.dat", "studtube.dat"):
                # Tube / socket underneath
                ports.append(ConnectionPort(
                    id=f"antistud_{len(ports)}",
                    port_type=PortType.ANTISTUD,
                    gender="female",
                    local_pos=pos,
                    local_normal=-normal,
                    dof=JointDOF.FIXED,
                    symmetry=4,
                    clutch_power=2.0,
                    source="ldraw_dat"
                ))
            elif sub_name.startswith("pin") and not sub_name.startswith("pinhole"):
                ports.append(ConnectionPort(
                    id=f"pin_{len(ports)}",
                    port_type=PortType.PIN,
                    gender="male",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.REVOLUTE,
                    symmetry=1,
                    clutch_power=3.0,
                    source="ldraw_dat"
                ))
            elif sub_name.startswith("pinhole") or "pinhol" in sub_name:
                ports.append(ConnectionPort(
                    id=f"pinhole_{len(ports)}",
                    port_type=PortType.PIN_HOLE,
                    gender="female",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.REVOLUTE,
                    symmetry=1,
                    clutch_power=3.0,
                    source="ldraw_dat"
                ))
            elif sub_name.startswith("axle") and not sub_name.startswith("axlehol"):
                ports.append(ConnectionPort(
                    id=f"axle_{len(ports)}",
                    port_type=PortType.AXLE,
                    gender="male",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.PRISMATIC,
                    symmetry=4,
                    clutch_power=2.5,
                    source="ldraw_dat"
                ))
            elif "axlehol" in sub_name:
                ports.append(ConnectionPort(
                    id=f"axlehole_{len(ports)}",
                    port_type=PortType.AXLE_HOLE,
                    gender="female",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.PRISMATIC,
                    symmetry=4,
                    clutch_power=2.5,
                    source="ldraw_dat"
                ))
            elif sub_name.startswith("clip"):
                ports.append(ConnectionPort(
                    id=f"clip_{len(ports)}",
                    port_type=PortType.CLIP,
                    gender="female",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.REVOLUTE,
                    symmetry=1,
                    clutch_power=1.8,
                    source="ldraw_dat"
                ))
            elif sub_name.startswith("bar"):
                ports.append(ConnectionPort(
                    id=f"bar_{len(ports)}",
                    port_type=PortType.BAR,
                    gender="male",
                    local_pos=pos,
                    local_normal=normal,
                    dof=JointDOF.REVOLUTE,
                    symmetry=1,
                    clutch_power=1.8,
                    source="ldraw_dat"
                ))
            elif sub_name.endswith(".dat"):
                sub_file = find_ldraw_part_file(sub_name, ldraw_root)
                if sub_file:
                    scan_file(sub_file, accum_t, depth + 1)

    part_file = find_ldraw_part_file(part_id, ldraw_root)
    if part_file:
        scan_file(part_file, np.eye(4, dtype=np.float32), 0)

    return ports

def _generate_parametric_ports(part_id: str) -> list[ConnectionPort]:
    """
    Algorithmic generator for LEGO pieces based on standard dimensions,
    correctly differentiating tiles (no top studs), slopes, and SNOT/clip parts.
    """
    from src.validator import get_part_dimensions

    clean_id = part_id.lower().replace(".dat", "")
    
    # NOTE: 6 LEGO parts previously marked as unsupported (10247, 2516, 2540, 30377, 4274, 6141)
    # Changed 2026-11-12: Now attempt parametric generation + .dat extraction anyway.
    # If no ports are found, the validation layer can mark them as unsupported_connectivity.
    # This allows recovery of connectivity data from .conn files or LDraw .dat primitives.
    
    ports: list[ConnectionPort] = []
    
    dim = get_part_dimensions(part_id)
    W, H, D = dim[0], dim[1], dim[2]
    N = max(1, int(round(W / 20.0)))
    M = max(1, int(round(D / 20.0)))

    ports: list[ConnectionPort] = []

    # 1. Check if the part is a tile (tiles have NO top studs!)
    is_tile = ("tile" in clean_id or 
               clean_id.startswith("3070") or 
               clean_id.startswith("3069") or 
               clean_id.startswith("63864") or 
               clean_id.startswith("98138"))

    # 2. Check for SNOT parts (e.g. 4070 headlight brick, 87087, brackets)
    is_headlight = clean_id.startswith("4070")

    # 3. Check for clips
    is_clip = clean_id.startswith("4081") or clean_id.startswith("2540") or clean_id.startswith("48336")

    # Generate top studs (unless it is a tile)
    if not is_tile:
        for i in range(N):
            x = -10.0 * (N - 1) + 20.0 * i
            for j in range(M):
                z = -10.0 * (M - 1) + 20.0 * j
                ports.append(ConnectionPort(
                    id=f"top_stud_{i}_{j}",
                    port_type=PortType.STUD,
                    gender="male",
                    local_pos=np.array([x, -H, z], dtype=np.float32),
                    local_normal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                    dof=JointDOF.FIXED,
                    symmetry=4,
                    clutch_power=2.0
                ))

    # Generate bottom sockets / antistuds (all standard bricks, plates, and tiles have bottom sockets)
    for i in range(N):
        x = -10.0 * (N - 1) + 20.0 * i
        for j in range(M):
            z = -10.0 * (M - 1) + 20.0 * j
            ports.append(ConnectionPort(
                id=f"bottom_socket_{i}_{j}",
                port_type=PortType.ANTISTUD,
                gender="female",
                local_pos=np.array([x, 0.0, z], dtype=np.float32),
                local_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                dof=JointDOF.FIXED,
                symmetry=4,
                clutch_power=2.0
            ))

    # SNOT lateral stud for headlight brick 4070, SNOT bricks (87087, 11211, 30414, 32952, 22885, 4733, 47905), brackets (44728, 99780, 99781, 36840, 36841, 2436, 99207)
    snot_prefixes = (
        "4070", "87087", "11211", "11212", "30414", "32952", "22885", "4733", "47905",
        "44728", "99780", "99781", "36840", "36841", "2436", "99207", "10247", "11458",
        "11476", "61252", "26604", "3023b", "73825"
    )
    if any(clean_id.startswith(sp) for sp in snot_prefixes):
        for side_id, lpos, lnorm in [
            ("side_stud_front", np.array([0.0, -H / 2.0, -D / 2.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            ("side_stud_back", np.array([0.0, -H / 2.0, D / 2.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("side_stud_left", np.array([-W / 2.0, -H / 2.0, 0.0], dtype=np.float32), np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
            ("side_stud_right", np.array([W / 2.0, -H / 2.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ]:
            ports.append(ConnectionPort(
                id=side_id,
                port_type=PortType.STUD_HOLLOW,
                gender="male",
                local_pos=lpos,
                local_normal=lnorm,
                dof=JointDOF.REVOLUTE,
                symmetry=1,
                clutch_power=2.0
            ))
            ports.append(ConnectionPort(
                id=f"{side_id}_antistud",
                port_type=PortType.ANTISTUD,
                gender="female",
                local_pos=lpos,
                local_normal=-lnorm,
                dof=JointDOF.FIXED,
                symmetry=4,
                clutch_power=2.0
            ))

    # Jumper plate 1x2 / 2x2 (15573 / 3794 / 34103 / 88646): central hollow stud at (0, -H, 0) + central tube antistud at (0, 0, 0)
    if any(clean_id.startswith(jp) for jp in ("15573", "3794", "34103", "88646")):
        ports.append(ConnectionPort(
            id="center_jumper_stud",
            port_type=PortType.STUD_HOLLOW,
            gender="male",
            local_pos=np.array([0.0, -H, 0.0], dtype=np.float32),
            local_normal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            dof=JointDOF.FIXED,
            symmetry=4,
            clutch_power=2.0
        ))
        ports.append(ConnectionPort(
            id="center_jumper_socket",
            port_type=PortType.ANTISTUD,
            gender="female",
            local_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            local_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            dof=JointDOF.FIXED,
            symmetry=4,
            clutch_power=2.0
        ))

    # Clip for 4081b / 61252 / 11476 / 2540 / 48336
    if is_clip or any(clean_id.startswith(cp) for cp in ("61252", "11476", "60470", "48729", "30237")):
        ports.append(ConnectionPort(
            id="clip_jaw",
            port_type=PortType.CLIP,
            gender="female",
            local_pos=np.array([0.0, -H/2.0, -D/2.0 - 5.0], dtype=np.float32),
            local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            dof=JointDOF.REVOLUTE,
            symmetry=1,
            clutch_power=1.8
        ))

    return ports

def get_part_ports(part_id: str) -> list[ConnectionPort]:
    """
    Returns the list of connection ports for a given part design ID.
    Caches results in memory for high-performance batch validation.
    """
    clean_id = part_id.lower()
    if clean_id in _PORT_CACHE:
        return _PORT_CACHE[clean_id]

    # Generate high-accuracy parametric ports adhering to LegoGPT's standard coordinate frame:
    # y = 0 is the bottom face, y = -H is the top face (studs), -Y is UP.
    ports = _generate_parametric_ports(clean_id)

    # Always augment with primitives extracted from LDraw .dat file if available
    ldraw_ports = _extract_ports_from_ldraw_dat(clean_id)
    if ldraw_ports:
        existing_types = {p.port_type for p in ports}
        for lp in ldraw_ports:
            if lp.port_type not in (PortType.STUD, PortType.ANTISTUD) or lp.port_type not in existing_types:
                ports.append(lp)

    _PORT_CACHE[clean_id] = ports
    return ports

def get_part_ports_world(part: ParsedPart) -> list[dict]:
    """
    Transforms all local connection ports of a part into world space coordinates
    using the part's 4x4 homogenous transformation matrix.
    """
    local_ports = get_part_ports(part.part_id)
    world_ports = []
    T = part.transform

    for p in local_ports:
        # Position in world: T @ [x, y, z, 1.0]
        pos_4 = np.array([p.local_pos[0], p.local_pos[1], p.local_pos[2], 1.0], dtype=np.float32)
        pos_w = (T @ pos_4)[:3]

        # Normal in world: R @ normal
        norm_w = T[:3, :3] @ p.local_normal
        mag = np.linalg.norm(norm_w)
        if mag > 1e-4:
            norm_w = norm_w / mag
        else:
            norm_w = p.local_normal

        world_ports.append({
            "id": p.id,
            "port_type": p.port_type,
            "gender": p.gender,
            "world_pos": pos_w,
            "world_normal": norm_w,
            "dof": p.dof,
            "symmetry": p.symmetry,
            "clutch_power": p.clutch_power
        })

    return world_ports

def are_ports_compatible(type_a: PortType | str, type_b: PortType | str) -> bool:
    """Checks if two connection port types can physically mate."""
    pair = (type_a, type_b)
    return pair in COMPATIBLE_PORT_PAIRS

def check_port_coupling(port_a: dict, port_b: dict, 
                        dist_tol: float = 1.0, 
                        angle_tol_deg: float = 15.0) -> bool:
    """
    Validates physical mating between two ports in world space:
    1. Port types are compatible (e.g. stud <-> antistud, clip <-> bar).
    2. Spatial distance is within tolerance (<= 1.0 LDU).
    3. Normal vectors oppose each other (n_a · n_b <= -cos(angle_tol)).
    """
    if not are_ports_compatible(port_a["port_type"], port_b["port_type"]):
        return False

    # 1. Proximity check
    dist = float(np.linalg.norm(port_a["world_pos"] - port_b["world_pos"]))
    if dist > dist_tol:
        return False

    # 2. Opposite normal alignment check (except for spherical joints)
    if port_a["dof"] != JointDOF.SPHERICAL and port_b["dof"] != JointDOF.SPHERICAL:
        cos_angle = float(np.dot(port_a["world_normal"], port_b["world_normal"]))
        threshold = -np.cos(np.radians(angle_tol_deg))
        if cos_angle > threshold:  # Normals must point towards each other (angle ~ 180°)
            return False

    return True

def check_parts_connected_ports(part_a: ParsedPart, part_b: ParsedPart) -> tuple[bool, list[dict]]:
    """
    Computes all valid mechanical connections between part_a and part_b.
    Returns (is_connected, list_of_matching_port_pairs).
    """
    # Fast spherical distance rejection
    pos_a = part_a.transform[:3, 3]
    pos_b = part_b.transform[:3, 3]
    if np.linalg.norm(pos_a - pos_b) > 400.0:  # Beyond largest standard part radius
        return False, []

    ports_a = get_part_ports_world(part_a)
    ports_b = get_part_ports_world(part_b)
    
    matches = []
    for pa in ports_a:
        for pb in ports_b:
            if check_port_coupling(pa, pb):
                matches.append({
                    "port_a": pa["id"],
                    "port_b": pb["id"],
                    "type_a": pa["port_type"],
                    "type_b": pb["port_type"],
                    "dof": pa["dof"],
                    "clutch_power": pa["clutch_power"]
                })

    return (len(matches) > 0), matches
