# GILWM reproduction guide

This document describes the public training and inference path for the single released **GILWM** checkpoint. The WorldArena submission variants associated with this work reuse the same checkpoint and change inference or post-processing; they are not separate released models.

## Release scope

The model repository contains one consolidated EMA state dict:

| File | Format | Size | SHA-256 |
| --- | --- | ---: | --- |
| `model_ema_bf16.pt` | bf16 PyTorch EMA state dict | 4,123,711,146 bytes | `fe29928334732c20241c452e6d4144e5293c2c6db549a604f390ac07a8d1fa67` |

Optimizer, scheduler, trainer state, datasets, condition caches, and historical initialization checkpoints are not bundled. The EMA can be used for inference or weight initialization, but it cannot resume the original optimizer state.

## Validated runtime

The GILWM launchers were validated on:

- Linux with Moore Threads MUSA;
- Moore Threads MTT S5000 80 GB accelerators;
- PyTorch with `torch_musa`;
- MCCL and FSDP2;
- Python 3.10;
- Docker with a site-provided MUSA runtime image.

The recorded Python package snapshot is in `projects/sam3d/configs/cosmos-musa-pip-freeze.txt`. CUDA and macOS execution have not been validated for this branch.

The runtime workspace is expected to provide:

```text
runtime-workspace/
  envs/cosmos-musa/bin/python
  weights/cosmos/models/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth
  weights/sam3/sam3.pt
  third_party/sam3-musa/
```

You must separately provision the compatible Cosmos Predict2.5 text encoder and VAE, SAM 3, SAM 3D Objects, DINOv2, and MoGe assets under their respective licenses.

## Download the model

```bash
hf auth login
hf download kurisu0306/GILWM model_ema_bf16.pt \
  --local-dir checkpoints/GILWM
```

Verify the downloaded file before use:

```bash
shasum -a 256 checkpoints/GILWM/model_ema_bf16.pt
```

## Data and condition cache

Training uses a `manifest.jsonl`. Each line must resolve a video, instruction, and sample directory:

```json
{"batch":"core15k","question_id":"core15k_00000001","video":"core15k/core15k_00000001/video.mp4","instruction":"core15k/core15k_00000001/instruction.json","sample_dir":"core15k/core15k_00000001"}
```

Each sample also has a fixed-shape `condition.pt` containing first-frame structure:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `sam3d_tokens` | `[8, 256, 768]` | Frozen teacher tokens used by optional representation alignment |
| `sam_masks` | `[8, 120, 160]` | SAM 3 instance masks |
| `sam_mask_meta` | `[8, 8]` | Boxes, confidence, area, prompt index, and valid bit |
| `sam3d_geometry` | `[8, 120, 160]` | Depth, XYZ, normals, and confidence |
| `sam3d_shape_latents` | `[8, 256, 8]` | SAM 3D Objects shape latents |
| `sam3d_object_pose` | `[8, 10]` | Translation, quaternion, and scale |

Use the preparation commands in the main README. The native SAM 3D Objects stage can run on a separate CUDA machine; copy the resulting sidecars back before merging the cache.

## Training

Copy and edit the public environment template:

```bash
cp projects/sam3d/configs/release.env.example .env.gilwm
source .env.gilwm
```

The portable launcher is `projects/sam3d/train_v93.sh`. The reference topology uses four nodes, eight devices per node, batch size four per device, 93 uniformly sampled frames, 640 × 480 resolution, AdamW, learning rate `3e-5`, 500 warm-up steps, and a 5,000-step cosine cycle.

Run one process launcher on each node:

```bash
export MASTER_ADDR=<rank-0-host>
export MASTER_PORT=29932
export NNODES=4
export NPROC_PER_NODE=8
export FSDP_SHARD_SIZE=8
export PER_DEVICE_BATCH_SIZE=4
export MAX_ITER=5000
export SAVE_ITER=500
export SAM3D_RUN_NAME=my_gilwm_run

bash projects/sam3d/train_v93.sh "$NODE_RANK"
```

`SAM3D_BASE_CHECKPOINT` initializes the SAM3D adapter. `SAM3D_COSMOS_OVERLAY_CHECKPOINT`, when set, then overlays matching Cosmos backbone parameters. Supplying only a different public base checkpoint produces a valid new run, but does not exactly reproduce the released model's historical initialization or data mixture.

The repository also retains temporal-action research code and its tests. No action-conditioned checkpoint is included in the GILWM model release.

## Inference

Create a request from `projects/sam3d/configs/inference.example.json`. Required inputs are a first frame, text instruction, and matching `condition.pt`:

```json
{
  "inference_type": "image2world",
  "name": "gilwm_example",
  "input_path": "/path/to/first_frame.png",
  "sam3d_condition_path": "/path/to/condition.pt",
  "prompt": "Move the cup onto the green plate.",
  "resolution": "480,640",
  "num_output_frames": 93,
  "fps": 24,
  "num_steps": 35,
  "seed": 0,
  "guidance": 7,
  "enable_autoregressive": false
}
```

Run inference:

```bash
export COSMOS_PROJECT="$PWD"
export COSMOS_SAM3D_WORKSPACE=/path/to/runtime-workspace
export COSMOS_MUSA_IMAGE=<your-musa-runtime-image>
export GILWM_INFERENCE_JSON=/path/to/gilwm_input.json
export GILWM_INFERENCE_OUTPUT=/path/to/output

bash projects/sam3d/scripts/run_gilwm_inference.sh checkpoints/GILWM
```

The launcher validates the generated video's resolution and frame count. For paths outside the repository or workspace, set `COSMOS_EXTRA_BIND_PATHS` to a colon-separated list of additional host paths. The launcher adds the request, input, cache, and output directories automatically.

For longer output, enable autoregressive inference and keep `chunk_size` at 93. With overlap one, two complete chunks contain 185 unique frames; another requested length is a truncated prefix and should be reported explicitly.

## Checks

The lightweight checks do not load the full model:

```bash
python tests/sam3d/test_sam3d_conditioned_dit.py
python tests/sam3d/test_sam3d_repa_loss.py
python projects/sam3d/scripts/test_merge_sam3d_object_sidecars.py
bash -n projects/sam3d/scripts/run_gilwm_inference.sh
```

A full inference test requires the MUSA runtime and all external model assets described above.
