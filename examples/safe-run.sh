#!/usr/bin/env bash
# Run one cooperative GPU command without selecting a device by hand.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  printf 'usage: %s COMMAND [ARG ...]\n' "$0" >&2
  exit 64
fi

if ! command -v gpu-steward >/dev/null 2>&1; then
  printf 'gpu-steward is not installed on this host\n' >&2
  exit 127
fi

gpu-steward doctor
status_json="$(gpu-steward status --json)"
printf '%s\n' "$status_json"

# Capacity is a request. GPU Steward chooses UUIDs and sets CUDA_VISIBLE_DEVICES.
exec gpu-steward run \
  --json \
  --min "${GPU_STEWARD_MIN:-1}" \
  --max "${GPU_STEWARD_MAX:-auto}" \
  -- "$@"
