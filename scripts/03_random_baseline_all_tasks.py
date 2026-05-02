"""Phase 0 / Step D — Full random baseline on libero_object (all 10 tasks).

Establishes the floor success rate for the suite. Any trained policy
must beat this to count as having learned anything.

Results saved to outputs/baselines/random_libero_object.json so we can
load them when generating the final writeup table.
"""
import os
import json
import time
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# --- config ---
SUITE = "libero_object"
N_EPS_PER_TASK = 5         # 5 init-state seeds per task -> 50 total rollouts
MAX_STEPS = 200
SEED = 0
OUT_DIR = "outputs/baselines"
# ---------------

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

print("=" * 70)
print(f"Phase 0 / Step D — Random baseline on {SUITE} (all tasks)")
print("=" * 70)

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

task_suite = benchmark.get_benchmark_dict()[SUITE]()
n_tasks = task_suite.n_tasks
log(f"Suite: {SUITE}, tasks: {n_tasks}, episodes/task: {N_EPS_PER_TASK}")

per_task = []
total_succ, total_eps = 0, 0
overall_t0 = time.time()

for tid in range(n_tasks):
    task = task_suite.get_task(tid)
    bddl = task_suite.get_task_bddl_file_path(tid)
    init_states = task_suite.get_task_init_states(tid)

    log(f"\n[Task {tid}/{n_tasks-1}] {task.language!r}")
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=128, camera_widths=128)

    succ = 0
    for ep in range(N_EPS_PER_TASK):
        env.reset()
        env.set_init_state(init_states[ep])
        for step in range(MAX_STEPS):
            action = np.concatenate([
                np.random.uniform(-0.1, 0.1, size=6),
                np.random.choice([-1.0, 1.0], size=1),
            ])
            _, _, done, _ = env.step(action)
            if done:
                succ += 1
                break

    env.close()
    rate = succ / N_EPS_PER_TASK
    log(f"  -> {succ}/{N_EPS_PER_TASK} success ({rate:.0%})")
    per_task.append({
        "task_id": tid,
        "task_name": task.name,
        "language": task.language,
        "successes": succ,
        "episodes": N_EPS_PER_TASK,
        "success_rate": rate,
    })
    total_succ += succ
    total_eps += N_EPS_PER_TASK

dur = time.time() - overall_t0
overall = total_succ / total_eps

print("\n" + "=" * 70)
print(f"RANDOM BASELINE — {SUITE}")
print("=" * 70)
for r in per_task:
    print(f"  Task {r['task_id']:2d}  {r['successes']:2d}/{r['episodes']}  "
          f"({r['success_rate']:.0%})  {r['language']}")
print("-" * 70)
print(f"  Overall: {total_succ}/{total_eps}  ({overall:.1%})")
print(f"  Wall time: {dur/60:.1f} min")
print("=" * 70)

out = {
    "suite": SUITE,
    "policy": "random",
    "seed": SEED,
    "episodes_per_task": N_EPS_PER_TASK,
    "max_steps": MAX_STEPS,
    "per_task": per_task,
    "overall_success_rate": overall,
    "wall_time_seconds": dur,
}
with open(f"{OUT_DIR}/random_{SUITE}.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"Saved {OUT_DIR}/random_{SUITE}.json")
print("\nSTEP D PASSED")
