"""Phase 0 / Step C — Random-policy rollout on libero_object task 0.

Validates the full eval loop:
- Full-length episode (not just one step)
- Video recording from agentview
- Success-detection signal (reward / done)
- Clean shutdown

Expected success rate: ~0% (random actions can't solve manipulation).
What matters here is that the loop COMPLETES without crashing.
"""
import os
import time
import numpy as np
import imageio.v2 as imageio

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# --- config ---
SUITE = "libero_object"
TASK_ID = 0
N_EPISODES = 3
MAX_STEPS = 200          # LIBERO_object tasks are typically solvable in <300 steps
SEED = 0
VIDEO_DIR = "videos"
# ---------------

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

print("=" * 70)
print(f"Phase 0 / Step C — Random rollout on {SUITE}, task {TASK_ID}")
print("=" * 70)

os.makedirs(VIDEO_DIR, exist_ok=True)
np.random.seed(SEED)

log("Loading suite & building env...")
task_suite = benchmark.get_benchmark_dict()[SUITE]()
task = task_suite.get_task(TASK_ID)
bddl = task_suite.get_task_bddl_file_path(TASK_ID)
log(f"Task: {task.language!r}")

env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    camera_heights=128,
    camera_widths=128,
)

# LIBERO ships per-task initial states for reproducible eval
init_states = task_suite.get_task_init_states(TASK_ID)
log(f"Available init states: {len(init_states)}")

results = []
for ep in range(N_EPISODES):
    log(f"\n--- Episode {ep+1}/{N_EPISODES} ---")
    env.reset()
    obs = env.set_init_state(init_states[ep])

    frames = [np.flipud(obs["agentview_image"])]
    success = False
    ep_t0 = time.time()

    for step in range(MAX_STEPS):
        # Random 7-D action: 6 EEF deltas in [-0.1, 0.1], gripper in {-1, 1}
        action = np.concatenate([
            np.random.uniform(-0.1, 0.1, size=6),
            np.random.choice([-1.0, 1.0], size=1),
        ])
        obs, reward, done, info = env.step(action)
        frames.append(np.flipud(obs["agentview_image"]))
        if done:
            success = True
            break

    ep_dt = time.time() - ep_t0
    fps = (step + 1) / ep_dt
    log(f"Episode {ep+1}: steps={step+1}  success={success}  "
        f"final_reward={reward:.3f}  duration={ep_dt:.1f}s  ({fps:.1f} steps/s)")

    out = f"{VIDEO_DIR}/01_random_ep{ep+1}_success={success}.mp4"
    imageio.mimsave(out, frames, fps=20)
    log(f"Saved {out} ({len(frames)} frames)")
    results.append(success)

env.close()

print("\n" + "=" * 70)
print(f"RESULTS  random policy on {SUITE} task {TASK_ID}")
print(f"  Episodes:     {N_EPISODES}")
print(f"  Successes:    {sum(results)}")
print(f"  Success rate: {sum(results)/N_EPISODES:.1%}")
print("=" * 70)
print("STEP C PASSED" if True else "")
