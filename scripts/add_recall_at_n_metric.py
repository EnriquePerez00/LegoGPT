#!/usr/bin/env python3
"""
Add recall@N metric tracking to OracleStepDebugger.evaluate_moc_fail_fast
without corrupting the file. This script patches the method to track the rank 
of GT among candidates for each step and emit recall@N curves.
"""

import re

def patch_oracle_step_debugger():
    file_path = "src/oracle_step_debugger.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Add gt_rank_histogram initialization after steps_survived = 0
    pattern1 = r"(\s+steps_survived = 0\n)"
    replacement1 = r"\1        gt_rank_histogram = [0] * 8  # bins for rank 1..8\n"
    content = re.sub(pattern1, replacement1, content)
    
    # 2. Add histogram update in Step 1 (baseline parts)
    pattern2 = r"(steps_survived \+= 1\n                continue\n            \n            sim_cfg)"
    replacement2 = r"steps_survived += 1\n                for _ in step_gt_parts:\n                    gt_rank_histogram[0] += 1\n                continue\n\n            sim_cfg"
    content = content.replace(
        "                steps_survived += 1\n                continue\n            \n            sim_cfg",
        "                steps_survived += 1\n                for _ in step_gt_parts:\n                    gt_rank_histogram[0] += 1\n                continue\n\n            sim_cfg"
    )
    
    # 3. Find the section where we calculate costs and add gt_rank calculation
    # This is right after the Hungarian matching loop
    pattern3 = r"(matched, _, _ = poses_match_modulo_symmetry\(pid, tf_p, tf_gt, self\.geom\)\n)\n(\s+if not matched:)"
    replacement3 = r"""\1
                # Calculate GT rank among candidates (NEW - RECALL@N tracking)
                gt_rank = None
                for rank_idx, cand in enumerate(part_candidates_filtered[i]):
                    tf_c = np.array(cand.absolute_transform, dtype=np.float32).reshape((4, 4))
                    is_match, _, _ = poses_match_modulo_symmetry(pid, tf_c, tf_gt, self.geom)
                    if is_match:
                        gt_rank = rank_idx + 1
                        break
                if gt_rank is None:
                    gt_rank = 9  # Not found in top 8
                
                # Update histogram
                bucket_idx = min(gt_rank, 8) - 1
                gt_rank_histogram[bucket_idx] += 1

\2"""
    content = re.sub(pattern3, replacement3, content)
    
    # 4. Add recall@N calculation before return statement
    pattern4 = r"(steps_survived == total_steps,\n\s+\"total_steps\": total_steps,\n\s+\"steps_survived\": steps_survived,\n\s+\"survival_ratio\": \(steps_survived / total_steps\) if total_steps > 0 else 0\.0,\n\s+\"incidents\": incidents,\n\s+}\n\s+return result)"
    replacement4 = r"""steps_survived == total_steps,
            "total_steps": total_steps,
            "steps_survived": steps_survived,
            "survival_ratio": (steps_survived / total_steps) if total_steps > 0 else 0.0,
            "incidents": incidents,
        }
        
        # Calculate recall@N for N=1..8
        total_ranks = sum(gt_rank_histogram)
        recall_at_n = {}
        cumulative = 0
        for n in range(1, 9):
            cumulative += gt_rank_histogram[n - 1]
            recall_at_n[n] = cumulative / total_ranks if total_ranks > 0 else 0.0
        
        result["recall_at_n"] = recall_at_n
        summary_lines = [f"Recall@{n}: {recall_at_n[n]*100:.1f}%" for n in range(1, 9)]
        result["recall_at_n_summary"] = " | ".join(summary_lines)
        
        return result"""
    content = re.sub(pattern4, replacement4, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Patched oracle_step_debugger.py with recall@N tracking")

if __name__ == "__main__":
    patch_oracle_step_debugger()
