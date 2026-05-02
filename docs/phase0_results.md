# Phase 0 — Pipeline Validation

**Status:** Complete  
**Cost:** $0 (local laptop only)  
**Goal:** Prove the LIBERO eval loop works end-to-end before spending any GPU money on training.

## What we validated

| Check | How | Result |
|---|---|---|
| LeRobot 0.5.1 install | `import lerobot` | OK |
| LIBERO 0.1.0 install | `import libero` | OK |
| MuJoCo + EGL rendering | First-frame PNG | OK (Franka + tabletop scene) |
| LIBERO `input()` prompt | Pre-created `~/.libero/config.yaml` | Patched, no longer fires |
| Observation keys (this version) | Live print | `agentview_image`, `robot0_eye_in_hand_image` (128×128×3 uint8) |
| Action space | Live step | 7-D (6 EEF deltas + gripper) |
| Episode rollout | 200-step random rollout, 3 episodes | Completes cleanly, MP4 saved |
| Eval throughput | 50 eps × 200 steps | ~5 min on CPU/iGPU (no GPU needed for eval) |

## Random-policy baseline — `libero_object`

Per-task success rate on 5 init-state seeds × 200 steps. Uniform random
6-D EEF deltas in [-0.1, 0.1] plus binary gripper action.

| Task | Language | Success |
|---|---|---|
| 0 | alphabet soup → basket | 0/5 |
| 1 | cream cheese → basket | 0/5 |
| 2 | salad dressing → basket | 0/5 |
| 3 | bbq sauce → basket | 0/5 |
| 4 | ketchup → basket | 0/5 |
| 5 | tomato sauce → basket | 0/5 |
| 6 | butter → basket | 0/5 |
| 7 | milk → basket | 0/5 |
| 8 | chocolate pudding → basket | 0/5 |
| 9 | orange juice → basket | 0/5 |
| **Total** | | **0/50 (0.0%)** |

This is our floor. Any trained policy must beat this. SmolVLA paper reports
~85% on this suite — that's the ceiling we're aiming for.

## Reproduce

```bash
conda activate smolvla
python scripts/01_inspect_libero_raw.py < /dev/null
python scripts/02_random_rollout.py < /dev/null
python scripts/03_random_baseline_all_tasks.py < /dev/null
```

## What this unlocks

We can now spend money on training, knowing the eval pipeline that
will measure whether training worked is bulletproof.
