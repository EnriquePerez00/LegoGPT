import os
import numpy as np
import pytest

from src.parser import ParsedPart
from src.validator import check_collisions, check_connectivity_and_gravity
from src.batch_generator import (
    get_optimal_workers,
    check_memory_safety,
    parse_prompt_semantics,
    generate_candidate_sets_parallel,
    record_human_feedback,
    refine_model_from_feedback,
    FEEDBACK_FILE_PATH,
)


def _deserialize_parts(parts_json: list[dict]) -> list[ParsedPart]:
    out = []
    for p in parts_json:
        t = np.array(p["transform"], dtype=np.float32).reshape(4, 4)
        out.append(ParsedPart(
            part_id=p["part_id"],
            color=int(p["color"]),
            transform=t,
            step_id=int(p.get("step_id", 0)),
        ))
    return out


def test_hardware_optimization_helpers():
    workers = get_optimal_workers()
    cpu_total = os.cpu_count() or 4
    assert workers == max(1, cpu_total - 2)

    mem = check_memory_safety()
    assert "total_gb" in mem
    assert "available_gb" in mem
    assert mem["total_gb"] > 0


def test_parallel_vehicle_batch_generation(tmp_path):
    res = generate_candidate_sets_parallel(
        prompt="Coche deportivo rojo",
        target_num_pieces=12,
        num_sets=4,
        output_dir=str(tmp_path / "vehicles"),
    )
    assert res["status"] == "success"
    assert res["category"] == "vehicle"
    assert len(res["candidates"]) == 4

    for cand in res["candidates"]:
        parts = _deserialize_parts(cand["parts"])
        assert len(parts) == 12
        assert cand["metrics"]["physics_valid"] is True
        assert len(check_collisions(parts)) == 0
        assert check_connectivity_and_gravity(parts) is True

        # Ground rules: Y <= 0.0 (1.0 tolerance) and <= 4 ground pieces
        for p in parts:
            assert p.transform[1, 3] <= 1.0
        ground_count = sum(1 for p in parts if abs(p.transform[1, 3]) <= 1.0)
        assert 1 <= ground_count <= 4
        assert cand["metrics"]["symmetry_score"] >= 0.75


def test_parallel_animal_batch_generation(tmp_path):
    res = generate_candidate_sets_parallel(
        prompt="Perro marrón",
        target_num_pieces=11,
        num_sets=4,
        output_dir=str(tmp_path / "animals"),
    )
    assert res["status"] == "success"
    assert res["category"] == "animal"
    assert len(res["candidates"]) == 4

    for cand in res["candidates"]:
        parts = _deserialize_parts(cand["parts"])
        assert len(parts) == 11
        assert cand["metrics"]["physics_valid"] is True
        assert len(check_collisions(parts)) == 0
        assert check_connectivity_and_gravity(parts) is True
        ground_count = sum(1 for p in parts if abs(p.transform[1, 3]) <= 1.0)
        assert 1 <= ground_count <= 4
        assert cand["metrics"]["symmetry_score"] >= 0.75


def test_human_feedback_and_mps_refinement(tmp_path):
    res = generate_candidate_sets_parallel(
        prompt="Dinosaurio verde",
        target_num_pieces=8,
        num_sets=2,
        output_dir=str(tmp_path / "dinos"),
    )
    cand_approve = res["candidates"][0]
    cand_reject = res["candidates"][1]

    fb1 = record_human_feedback(
        candidate_id=cand_approve["candidate_id"],
        prompt="Dinosaurio verde",
        category="animal",
        approved=True,
        rating=5,
        parts=cand_approve["parts"],
    )
    assert fb1["status"] == "success"
    assert fb1["total_feedback"] >= 1

    fb2 = record_human_feedback(
        candidate_id=cand_reject["candidate_id"],
        prompt="Dinosaurio verde",
        category="animal",
        approved=False,
        rating=1,
        parts=cand_reject["parts"],
    )
    assert fb2["status"] == "success"

    refine_res = refine_model_from_feedback(epochs=4)
    assert refine_res["status"] == "success"
    assert len(refine_res["loss_history"]) == 4


def test_convertible_red_sports_car_100_pieces_6_candidates():
    """
    Internal stress & step-by-step verification test requested by USER:
    - Prompt: "coche deportivo de color rojo y descapotable"
    - Candidates: 6 sets (num_sets=6)
    - Target pieces per set: 100 pieces (target_num_pieces=100)
    - Validates every construction step (Step 0 -> Step K) and every individual piece (1..100)
      for stud-to-socket alignment, zero OBB overlaps, Y <= 0.0 surface rule, <= 4 ground parts,
      COM stability, and convertible red sports car morphology.
    """
    from src.validator import check_connection_optimized, get_part_dimensions
    from generative.llm_pipeline.rl_loop import calculate_com_stability

    prompt = "coche deportivo de color rojo y descapotable"
    target_pieces = 100
    num_sets = 6

    res = generate_candidate_sets_parallel(
        prompt=prompt,
        target_num_pieces=target_pieces,
        num_sets=num_sets,
        output_dir="outputs/batch_test_convertible_100",
    )

    assert res["status"] == "success"
    assert res["category"] == "vehicle"
    assert res["subtype"] == "convertible_sport"
    assert res["semantics"]["primary_color"] == 4  # Red (LDraw code 4)
    assert len(res["candidates"]) == num_sets

    print("\n" + "=" * 95)
    print(f"TEST INTERNO: '{prompt}' | {num_sets} Candidatos x {target_pieces} Piezas")
    print("=" * 95)
    hw = res["hardware"]
    print(
        f"[Hardware M4] Device: {hw['device'].upper()} | Workers: {hw['parallel_workers']} cores "
        f"| Tiempo Total: {hw['elapsed_seconds']}s ({hw['ms_per_candidate']} ms/set) "
        f"| RAM libre: {hw['memory']['available_gb']} GB / {hw['memory']['total_gb']} GB ({hw['memory']['percent_used']}% usado)"
    )
    print("-" * 95)

    for idx, cand in enumerate(res["candidates"], start=1):
        parts = _deserialize_parts(cand["parts"])
        assert len(parts) == target_pieces, f"Candidato #{idx} tiene {len(parts)} piezas en vez de {target_pieces}"

        # Group parts by step_id to verify step-by-step assembly sequence without errors
        steps_map: dict[int, list[ParsedPart]] = {}
        for p in parts:
            steps_map.setdefault(p.step_id, []).append(p)

        ordered_step_ids = sorted(steps_map.keys())
        cumulative_parts: list[ParsedPart] = []
        step_errors: list[str] = []

        for s_idx, sid in enumerate(ordered_step_ids):
            step_parts = steps_map[sid]
            for sp in step_parts:
                # 1. Surface rule: Y <= 0.0 (1.0 tolerance)
                y_val = float(sp.transform[1, 3])
                if y_val > 1.0:
                    step_errors.append(f"Step {sid}: Pieza {sp.part_id} bajo suelo Y={y_val}")

                # 2. If not on ground (step > 0), must have stud-to-socket connection to existing cumulative_parts
                if s_idx > 0:
                    connected_to_prev = any(check_connection_optimized(sp, prev_p) for prev_p in cumulative_parts)
                    if not connected_to_prev:
                        step_errors.append(f"Step {sid}: Pieza {sp.part_id} sin encaje stud-to-socket con estructura previa")

                cumulative_parts.append(sp)

            # 3. Ground contact count at every step <= 4
            g_count = sum(1 for cp in cumulative_parts if abs(cp.transform[1, 3]) <= 1.0)
            if not (1 <= g_count <= 4):
                step_errors.append(f"Step {sid}: ground_parts={g_count} (violación de límite <= 4)")

            # 4. COM stability at every step once chassis bridges wheels (from Step 1 onwards)
            if s_idx >= 1 and not calculate_com_stability(cumulative_parts):
                step_errors.append(f"Step {sid}: Centro de masa (COM) inestable")

        # 5. Full connectivity & OBB / Trimesh collision check across all 100 pieces
        assert check_connectivity_and_gravity(parts) is True, f"Candidato #{idx} grafo desconectado"
        collisions = check_collisions(parts)
        assert len(collisions) == 0, f"Candidato #{idx} presenta colisiones OBB: {collisions}"
        assert len(step_errors) == 0, f"Candidato #{idx} errores paso a paso: {step_errors}"
        assert cand["metrics"]["physics_valid"] is True

        # Color and convertible morphology checks
        red_count = sum(1 for p in parts if p.color == 4)
        windscreen_count = sum(1 for p in parts if p.part_id == "3823.dat")
        wheel_count = sum(1 for p in parts if p.part_id in ("30027.dat", "42610.dat", "3139.dat"))
        cockpit_roof_parts = sum(
            1 for p in parts
            if p.transform[1, 3] < -48.0 and abs(p.transform[0, 3]) <= 15.0 and -15.0 <= p.transform[2, 3] <= 20.0
        )

        assert wheel_count == 4, f"Candidato #{idx} debe tener 4 ruedas, tiene {wheel_count}"
        assert windscreen_count >= 1, f"Candidato #{idx} debe tener parabrisas delantero"
        assert cockpit_roof_parts == 0, f"Candidato #{idx} es descapotable pero tiene {cockpit_roof_parts} piezas de techo sobre cabina"
        assert red_count >= 75, f"Candidato #{idx} debe ser predominantemente rojo (>=75%), tiene {red_count}/100"

        m = cand["metrics"]
        print(
            f"Candidato #{idx} [{cand['candidate_id']}] | Piezas: {m['num_pieces']}/100 | Pasos: {len(ordered_step_ids)} "
            f"| Física OK: {m['physics_valid']} | Colisiones: 0 | Ground: {m['ground_parts']}/4 "
            f"| Rojas: {red_count}% | Simetría: {int(m['symmetry_score']*100)}% "
            f"| Interlocking: {m['interlocking_score']} | Score Total: {m['total_score']} "
            f"| LDraw: {cand['ldr_path']}"
        )
        # Print first 5 steps and last 2 steps of Candidate #1 to show exact step-by-step trace
        if idx == 1:
            print("  └─ Desglose Paso a Paso (Muestra Candidato #1):")
            sample_steps = ordered_step_ids[:5] + ordered_step_ids[-2:]
            for sid in sample_steps:
                sp_list = steps_map[sid]
                desc = ", ".join(
                    f"{p.part_id}(col={p.color}, pos=[{p.transform[0,3]:.0f},{p.transform[1,3]:.0f},{p.transform[2,3]:.0f}])"
                    for p in sp_list
                )
                print(f"     • Step {sid:02d} ({len(sp_list)} pzs): {desc}")

    print("=" * 95 + "\n")

