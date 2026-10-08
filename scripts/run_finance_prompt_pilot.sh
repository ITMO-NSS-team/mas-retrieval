#!/bin/sh
# Run from the repository root on the prepared Linux experiment machine.
# Six technical-pilot answers; this is not one of the full 150-question repeats.
set -eu
exec env JUDGE_MODEL=openai/gpt-4o-mini HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  just run --benchmark financebench \
  --systems adas_compact_prompt generated_single_agent_legacy \
  --model openai/gpt-4o-mini --generation-mode one_time \
  --adapter-config configs/finance_prompt_ablation.json \
  --sample-n 3 --repeats 1 --condition-id finance_prompt_ablation_pilot_v1 \
  --note "P006 technical pilot: original execution policies, passive accounting, 3 questions per system"
