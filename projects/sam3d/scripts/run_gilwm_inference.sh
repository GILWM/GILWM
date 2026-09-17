#!/usr/bin/env bash
set -euo pipefail

checkpoint_arg=${1:?usage: run_gilwm_inference.sh CHECKPOINT_DIR}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=${COSMOS_PROJECT:-$(git -C "${script_dir}" rev-parse --show-toplevel)}
workspace=${COSMOS_SAM3D_WORKSPACE:?set COSMOS_SAM3D_WORKSPACE}
input_arg=${GILWM_INFERENCE_JSON:?set GILWM_INFERENCE_JSON to a JSON or JSONL request}
output_dir=${GILWM_INFERENCE_OUTPUT:-${workspace}/inference-outputs/gilwm}
checkpoint_dir=$(cd "${checkpoint_arg}" && pwd)
input_json=$(cd "$(dirname "${input_arg}")" && pwd)/$(basename "${input_arg}")
model_pt=${checkpoint_dir}/model_ema_bf16.pt
runner=${repo_root}/projects/sam3d/scripts/run_cosmos_sam3d_musa.sh
verifier=${repo_root}/projects/sam3d/scripts/verify_video_output.py
inference_gpus=${GILWM_INFERENCE_GPUS:-2}
inference_devices=${GILWM_INFERENCE_DEVICES:-0,1}

for required in "${model_pt}" "${input_json}" "${runner}" "${verifier}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required GILWM inference artifact: ${required}" >&2
    exit 2
  fi
done

readarray -t request_meta < <(
  python3 -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
payload = json.loads(text) if path.suffix == ".json" else json.loads(next(line for line in text.splitlines() if line.strip()))
print(payload["name"])
print(payload.get("num_output_frames", 93))
base = path.parent
for key in ("input_path", "sam3d_condition_path"):
    value = pathlib.Path(payload[key])
    print((value if value.is_absolute() else base / value).resolve())
' "${input_json}"
)
sample_name=${request_meta[0]}
expected_frames=${request_meta[1]}
input_path=${request_meta[2]}
condition_path=${request_meta[3]}

mkdir -p "${output_dir}"
output_dir=$(cd "${output_dir}" && pwd)
for required in "${input_path}" "${condition_path}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required GILWM input: ${required}" >&2
    exit 2
  fi
done
export COSMOS_PROJECT="${repo_root}"
export SAM3D_BASE_CHECKPOINT="${model_pt}"
export MUSA_VISIBLE_DEVICES="${inference_devices}"

bind_paths=(
  "$(cd "$(dirname "${input_json}")" && pwd)"
  "$(cd "$(dirname "${input_path}")" && pwd)"
  "$(cd "$(dirname "${condition_path}")" && pwd)"
  "$(cd "${output_dir}" && pwd)"
)
if [[ -n "${COSMOS_EXTRA_BIND_PATHS:-}" ]]; then
  bind_paths+=("${COSMOS_EXTRA_BIND_PATHS}")
fi
export COSMOS_EXTRA_BIND_PATHS=$(IFS=:; echo "${bind_paths[*]}")

"${runner}" \
  -m torch.distributed.run --standalone --nproc-per-node="${inference_gpus}" \
  examples/inference.py \
  -i "${input_json}" \
  -o "${output_dir}" \
  --checkpoint-path "${model_pt}" \
  --experiment predict2_video2world_inference_2b_sam3d \
  --disable-guardrails

video=${output_dir}/${sample_name}.mp4
if [[ ! -s "${video}" ]]; then
  echo "Inference did not produce the expected video: ${video}" >&2
  exit 3
fi

"${runner}" "${verifier}" "${video}" --width 640 --height 480 --frames "${expected_frames}"
echo "GILWM_INFERENCE_OK ${video} width=640 height=480 frames=${expected_frames}"
