#!/bin/bash
# Multi-concurrency inference — full 8-GPU single-machine, data-parallel across replicas.
# Can be executed from anywhere.
#
# Verified config for Qwen3-4B-Instruct on 8x RTX 3090 (24GB):
#   nnodes=1  n_gpus=8  tp=2  →  dp = 8/2 = 4 vLLM replicas
#   gpu_memory_utilization=0.9 (set in parallel_infer.py, official default)
#   max_num_seqs=64            (3090 24GB; raise to 256 on large-VRAM GPUs)
#   concurrency=64             (in agent_config_localdocker.yaml, official default)
#
# Usage:
#   bash scripts/infer_multi.sh [MODEL_PATH] [DATA_PATH] [AGENT_CONFIG]
#   bash scripts/infer_multi.sh /data/models/Qwen3-4B
#   MAX_SAMPLES=10 NWORKERS=8 TP=2 bash scripts/infer_multi.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

MODEL_PATH=${1:-/path/to/Qwen3-4B}
DATA_PATH=${2:-${SCRIPT_DIR}/swe_bench_verified_modal.parquet}
AGENT_CONFIG=${3:-${SCRIPT_DIR}/agent_config_localdocker.yaml}

# Optional: enable KVCAware llm-router (plugin_extension). Unset = built-in router.
# e.g. ROUTER_CONFIG=pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml
ROUTER_CONFIG=${ROUTER_CONFIG:-}

# ── Tunables (env-overridable) ──
# Single machine with N GPUs. TP must divide N; dp = N / TP.
NWORKERS=${NWORKERS:-8}          # agent rollout workers
NNODES=${NNODES:-1}              # physical nodes (147 is single-machine → 1, NOT 4)
NGPUS=${NGPUS:-8}                # GPUs per node
TP=${TP:-2}                      # tensor parallel; dp = NGPUS*NNODES / TP
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64} # per-replica vLLM concurrency; 256 on large-VRAM, 64 on 24GB
MAX_TURNS=${MAX_TURNS:-100}
N=${N:-1}                # rollouts per prompt; set N>1 to repeat prompts for prefix cache testing

MAX_SAMPLES=${MAX_SAMPLES:--1}   # -1 = full dataset
PROMPT_LEN=${PROMPT_LEN:-32768}
RESPONSE_LEN=${RESPONSE_LEN:-65536}

# Use all GPUs by default (override with CUDA_VISIBLE_DEVICES=N,N,... if shared machine)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ── Pre-flight checks ──
for var_name in DATA_PATH MODEL_PATH AGENT_CONFIG; do
    path="${!var_name}"
    if [ ! -e "$path" ]; then
        echo "ERROR: ${var_name} not found: ${path}" >&2
        exit 1
    fi
done

# sanity: TP must divide total GPUs
TOTAL_GPUS=$((NNODES * NGPUS))
if [ $((TOTAL_GPUS % TP)) -ne 0 ]; then
    echo "ERROR: tensor-parallel-size ($TP) must divide total GPUs ($TOTAL_GPUS)" >&2
    exit 1
fi
echo "=== infer_multi: $NNODES node x $NGPUS GPU, tp=$TP → dp=$((TOTAL_GPUS / TP)) replicas, max_num_seqs=$MAX_NUM_SEQS, workers=$NWORKERS, max_samples=$MAX_SAMPLES ==="

ROUTER_ARGS=()
if [ -n "$ROUTER_CONFIG" ]; then
    ROUTER_ARGS+=(--router-config-path "$ROUTER_CONFIG")
fi

python ${PROJECT_ROOT}/examples/agent_interaction/parallel_infer.py \
    --data-path $DATA_PATH \
    --model-path $MODEL_PATH \
    --agent-config-path $AGENT_CONFIG \
    --num-workers $NWORKERS \
    --max-turns $MAX_TURNS \
    --nnodes $NNODES \
    --n-gpus-per-node $NGPUS \
    --tensor-parallel-size $TP \
    --max-num-seqs $MAX_NUM_SEQS \
    --prompt-length $PROMPT_LEN \
    --response-length $RESPONSE_LEN \
    --max-samples $MAX_SAMPLES \
    --n $N \
    "${ROUTER_ARGS[@]}"
