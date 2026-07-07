import numpy as np
from src.parser import ParsedPart
from src.validator import get_studs_and_sockets_world, check_collisions

class LegoGraphValidator:
    def __init__(self, current_state=None):
        """
        Initializes the validator with the current construction state.
        
        Args:
            current_state: List of parts. Each part can be a ParsedPart object, 
                           a dictionary, or an object with attributes.
        """
        self.current_state = self._normalize_state(current_state)

    def _normalize_part(self, part) -> ParsedPart:
        """Normalizes various part formats into a standard ParsedPart object."""
        if isinstance(part, ParsedPart):
            return part
            
        if isinstance(part, dict):
            part_id = part.get("part_id") or part.get("id") or part.get("part_name") or ""
            color = part.get("color", 0)
            step_id = part.get("step_id", 0)
            
            if "transform" in part:
                transform = np.array(part["transform"], dtype=np.float32)
            else:
                pos = part.get("position") or part.get("pos") or [0.0, 0.0, 0.0]
                rot = part.get("rotation") or part.get("rot")
                
                transform = np.eye(4, dtype=np.float32)
                if rot is not None:
                    rot = np.array(rot, dtype=np.float32)
                    if rot.shape == (3, 3):
                        transform[:3, :3] = rot
                    elif rot.shape == (9,):
                        transform[:3, :3] = rot.reshape(3, 3)
                    elif rot.shape == (4, 4):
                        transform = rot
                transform[:3, 3] = pos
            return ParsedPart(part_id=part_id, color=color, transform=transform, step_id=step_id)
            
        # Fallback for object with attributes
        part_id = getattr(part, "part_id", getattr(part, "id", ""))
        color = getattr(part, "color", 0)
        step_id = getattr(part, "step_id", 0)
        
        if hasattr(part, "transform"):
            transform = np.array(part.transform, dtype=np.float32)
        else:
            pos = getattr(part, "position", getattr(part, "pos", [0.0, 0.0, 0.0]))
            rot = getattr(part, "rotation", getattr(part, "rot", None))
            transform = np.eye(4, dtype=np.float32)
            if rot is not None:
                rot = np.array(rot, dtype=np.float32)
                if rot.shape == (3, 3):
                    transform[:3, :3] = rot
                elif rot.shape == (9,):
                    transform[:3, :3] = rot.reshape(3, 3)
            transform[:3, 3] = pos
            
        return ParsedPart(part_id=part_id, color=color, transform=transform, step_id=step_id)

    def _normalize_state(self, state) -> list[ParsedPart]:
        """Converts a state list into a list of ParsedParts."""
        if state is None:
            return []
        return [self._normalize_part(p) for p in state]

    def can_place_brick(self, new_brick, current_state=None) -> bool:
        """
        Validates if new_brick can be physically placed in the construction.
        
        Checks:
        (a) That the new_brick does not physically overlap/intersect with any 
            existing part (using bounding boxes / trimesh collision engine).
        (b) That at least one stud of the new brick aligns with a socket (antistud) 
            of an existing part, or vice-versa, within a tolerance of 5 LDU.
            
        Args:
            new_brick: The brick to be placed.
            current_state: Optional list of parts. If provided, overrides the 
                           state passed in the constructor.
                           
        Returns:
            bool: True if the placement is valid, False otherwise.
        """
        normalized_new = self._normalize_part(new_brick)
        
        state_to_use = self.current_state
        if current_state is not None:
            state_to_use = self._normalize_state(current_state)
            
        # Rule (a): Collision checking
        # Check collision by checking if the index of new_brick collides with any other index
        parts = state_to_use + [normalized_new]
        new_idx = len(parts) - 1
        
        collisions = check_collisions(parts)
        for i, j in collisions:
            if i == new_idx or j == new_idx:
                # Intersection found
                return False
                
        # Rule (b): Connectivity checking
        # If the state is empty, placement is allowed (e.g. placing the first brick on the ground/base)
        if not state_to_use:
            return True
            
        studs_new, sockets_new = get_studs_and_sockets_world(normalized_new)
        
        has_connectivity = False
        tolerance = 5.0
        
        for existing_part in state_to_use:
            studs_exist, sockets_exist = get_studs_and_sockets_world(existing_part)
            
            # Check if any stud of the new brick connects to any socket/antistud of the existing part
            if len(studs_new) > 0 and len(sockets_exist) > 0:
                # Compute pairwise Euclidean distance between studs_new and sockets_exist
                dists = np.linalg.norm(studs_new[:, None, :] - sockets_exist[None, :, :], axis=2)
                if np.any(dists < tolerance):
                    has_connectivity = True
                    break
                    
            # Check if any socket/antistud of the new brick connects to any stud of the existing part
            if len(sockets_new) > 0 and len(studs_exist) > 0:
                dists = np.linalg.norm(sockets_new[:, None, :] - studs_exist[None, :, :], axis=2)
                if np.any(dists < tolerance):
                    has_connectivity = True
                    break
                    
        return has_connectivity
