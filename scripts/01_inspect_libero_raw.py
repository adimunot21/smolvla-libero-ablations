"""Phase 0 / Step B — Validate raw LIBERO sim works.

What this proves if it succeeds:
- MuJoCo + EGL rendering works (no GL context crashes)
- LIBERO assets are findable (BDDL files, object meshes)
- We see the exact observation keys this version emits
- No hidden input() prompts hang on first import
"""
import os
import sys
import time
import numpy as np

# Force EGL off-screen rendering — required for headless / single-display Linux
os.environ.setdefault("MUJOCO_GL", "egl")

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

print("=" * 70)
print("Phase 0 / Step B — LIBERO raw inspection")
print("=" * 70)

log("Importing libero...")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
log("Imports OK.")

log(f"Asset path:  {get_libero_path('assets')}")
log(f"BDDL path:   {get_libero_path('bddl_files')}")
log(f"Init states: {get_libero_path('init_states')}")

log("Loading benchmark dict...")
benchmark_dict = benchmark.get_benchmark_dict()
log(f"Available suites: {list(benchmark_dict.keys())}")

suite_name = "libero_object"
log(f"Loading suite: {suite_name}")
task_suite = benchmark_dict[suite_name]()
log(f"Suite has {task_suite.n_tasks} tasks.")

task_id = 0
task = task_suite.get_task(task_id)
task_bddl = task_suite.get_task_bddl_file_path(task_id)
log(f"Task 0: name='{task.name}'")
log(f"        language='{task.language}'")
log(f"        bddl='{task_bddl}'")

log("Building OffScreenRenderEnv (this initializes MuJoCo)...")
env = OffScreenRenderEnv(
    bddl_file_name=task_bddl,
    camera_heights=256,
    camera_widths=256,
)
log("Env built OK.")

log("Resetting env (renders first frame)...")
obs = env.reset()
log("Reset OK.")

print("\n" + "=" * 70)
print("OBSERVATION KEYS  (these are the camera/state names we need to match)")
print("=" * 70)
for k, v in sorted(obs.items()):
    if hasattr(v, "shape"):
        print(f"  {k:42s} shape={str(v.shape):20s} dtype={v.dtype}")
    else:
        print(f"  {k:42s} type={type(v).__name__}")

# Action probe — LIBERO uses 7-DoF (6 EEF delta + gripper)
action_dim = 7
random_action = np.random.uniform(-0.1, 0.1, size=(action_dim,))
log(f"\nStepping env with random {action_dim}-D action...")
obs2, reward, done, info = env.step(random_action)
log(f"Step OK. reward={reward:.4f}  done={done}")
if isinstance(info, dict):
    log(f"Info keys: {list(info.keys())}")

# Save the first frame so we can eyeball that rendering actually worked
img_key = next((k for k in obs if "agentview" in k.lower() and "image" in k.lower()), None)
if img_key:
    import imageio
    img = np.flipud(obs[img_key])  # LIBERO renders upside-down
    os.makedirs("videos", exist_ok=True)
    out = "videos/00_libero_first_frame.png"
    imageio.imwrite(out, img)
    log(f"Saved first frame: {out} (shape={img.shape}, range=[{img.min()},{img.max()}])")
else:
    log("WARNING: no agentview image key found — check obs keys above")

env.close()
log("Env closed cleanly.")
print("\n" + "=" * 70)
print("STEP B PASSED")
print("=" * 70)
