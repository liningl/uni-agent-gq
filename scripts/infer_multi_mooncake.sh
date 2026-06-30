#!/bin/bash
# Wrapper around infer_multi.sh that starts Mooncake and enables the KVCAware
# router with mooncake tier store.
#
# Usage:
#   bash scripts/infer_multi_mooncake.sh [MODEL_PATH] [DATA_PATH] [AGENT_CONFIG]
#
# Default mode is standalone-store: an external mooncake_client owns the CPU
# pool and FileStorage SSD tier, while vLLM ranks are pure requesters.
# Set MOONCAKE_MODE=embedded to use the previous per-vLLM-rank memory segments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH=${1:-/path/to/Qwen3-4B}
DATA_PATH=${2:-${SCRIPT_DIR}/swe_bench_verified_modal.parquet}
AGENT_CONFIG=${3:-${SCRIPT_DIR}/agent_config_localdocker.yaml}

MOONCAKE_RPC_PORT=${MOONCAKE_RPC_PORT:-50051}
MOONCAKE_HTTP_PORT=${MOONCAKE_HTTP_PORT:-8080}
MOONCAKE_METRICS_PORT=${MOONCAKE_METRICS_PORT:-19003}
MOONCAKE_MODE=${MOONCAKE_MODE:-standalone-store}
MOONCAKE_DRY_RUN=${MOONCAKE_DRY_RUN:-0}
MOONCAKE_LOG_DIR=${MOONCAKE_LOG_DIR:-}
TP=${TP:-2}

MOONCAKE_EMBEDDED_GLOBAL_SEGMENT_SIZE=${MOONCAKE_EMBEDDED_GLOBAL_SEGMENT_SIZE:-2147483648}
MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE:-2147483648}
MOONCAKE_OWNER_HOST=${MOONCAKE_OWNER_HOST:-127.0.0.1}
MOONCAKE_OWNER_SEGMENT_PORT=${MOONCAKE_OWNER_SEGMENT_PORT:-12353}
MOONCAKE_OWNER_RPC_PORT=${MOONCAKE_OWNER_RPC_PORT:-12453}
MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE=${MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE:-4 GB}
MOONCAKE_SSD_DIR=${MOONCAKE_SSD_DIR:-/tmp/mooncake_ssd}
MOONCAKE_SSD_LOCAL_BUFFER_SIZE=${MOONCAKE_SSD_LOCAL_BUFFER_SIZE:-67108864}
MOONCAKE_SSD_TOTAL_SIZE_LIMIT=${MOONCAKE_SSD_TOTAL_SIZE_LIMIT:-1073741824}
MOONCAKE_SSD_HEARTBEAT_INTERVAL=${MOONCAKE_SSD_HEARTBEAT_INTERVAL:-1}
MOONCAKE_OFFLOAD_ON_EVICT=${MOONCAKE_OFFLOAD_ON_EVICT:-0}

MOONCAKE_MODEL_NAME=${MOONCAKE_MODEL_NAME:-$(basename "${MODEL_PATH%/}")}

case "${MOONCAKE_MODE}" in
    standalone-store|embedded) ;;
    *)
        echo "[mooncake] ERROR: MOONCAKE_MODE must be standalone-store or embedded, got ${MOONCAKE_MODE}" >&2
        exit 1
        ;;
esac

echo "=== infer_multi_mooncake: model=${MOONCAKE_MODEL_NAME}, tp=${TP}, mode=${MOONCAKE_MODE}, rpc_port=${MOONCAKE_RPC_PORT} ==="

MOONCAKE_PID=""
MOONCAKE_OWNER_PID=""
MOONCAKE_CONFIG_FILE=$(mktemp /tmp/mooncake_config_XXXXXX.json)
MOONCAKE_MASTER_LOG=""
MOONCAKE_OWNER_LOG=""

cleanup() {
    if [ -n "${MOONCAKE_OWNER_PID:-}" ]; then
        kill "${MOONCAKE_OWNER_PID}" 2>/dev/null || true
    fi
    if [ -n "${MOONCAKE_PID:-}" ]; then
        kill "${MOONCAKE_PID}" 2>/dev/null || true
    fi
    rm -f "${MOONCAKE_CONFIG_FILE:-}"
}
trap cleanup EXIT

if [ "${MOONCAKE_MODE}" = "standalone-store" ]; then
    cat > "${MOONCAKE_CONFIG_FILE}" << JSON
{
    "mode": "standalone-store",
    "metadata_server": "P2PHANDSHAKE",
    "master_server_address": "127.0.0.1:${MOONCAKE_RPC_PORT}",
    "global_segment_size": 0,
    "local_buffer_size": ${MOONCAKE_LOCAL_BUFFER_SIZE},
    "protocol": "tcp",
    "device_name": "",
    "enable_offload": true
}
JSON
    export MOONCAKE_PREFERRED_SEGMENT=${MOONCAKE_PREFERRED_SEGMENT:-"${MOONCAKE_OWNER_HOST}:${MOONCAKE_OWNER_SEGMENT_PORT}"}
    export VLLM_MOONCAKE_STORE_TIER_LOG=${VLLM_MOONCAKE_STORE_TIER_LOG:-1}
else
    cat > "${MOONCAKE_CONFIG_FILE}" << JSON
{
    "mode": "embedded",
    "metadata_server": "P2PHANDSHAKE",
    "master_server_address": "127.0.0.1:${MOONCAKE_RPC_PORT}",
    "global_segment_size": ${MOONCAKE_EMBEDDED_GLOBAL_SEGMENT_SIZE},
    "local_buffer_size": ${MOONCAKE_LOCAL_BUFFER_SIZE},
    "protocol": "tcp",
    "device_name": "",
    "enable_offload": false
}
JSON
fi

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_FILE}"

# Required: hex block hashes (not int) for MooncakeTierStore mapping
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0

echo "[mooncake] Config written to ${MOONCAKE_CONFIG_FILE}"
if [ "${MOONCAKE_MODE}" = "standalone-store" ]; then
    echo "[mooncake] MOONCAKE_PREFERRED_SEGMENT=${MOONCAKE_PREFERRED_SEGMENT}"
fi
if [ -n "${MOONCAKE_LOG_DIR}" ]; then
    mkdir -p "${MOONCAKE_LOG_DIR}"
    MOONCAKE_MASTER_LOG="${MOONCAKE_LOG_DIR}/mooncake_master.log"
    MOONCAKE_OWNER_LOG="${MOONCAKE_LOG_DIR}/mooncake_client_owner.log"
    echo "[mooncake] MOONCAKE_LOG_DIR=${MOONCAKE_LOG_DIR}"
    echo "[mooncake] master.log=${MOONCAKE_MASTER_LOG}"
    echo "[mooncake] owner.log=${MOONCAKE_OWNER_LOG}"
fi

if [ "${MOONCAKE_DRY_RUN}" = "1" ]; then
    echo "[mooncake] Dry run enabled; skipping process startup and infer_multi.sh"
    echo "[mooncake] MOONCAKE_MODE=${MOONCAKE_MODE}"
    echo "[mooncake] MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
    cat "${MOONCAKE_CONFIG_PATH}"
    exit 0
fi

# ── Start mooncake_master ──
echo "[mooncake] Starting mooncake_master..."
MOONCAKE_MASTER_ARGS=(
    mooncake_master
    --enable_http_metadata_server=true
    --http_metadata_server_port="${MOONCAKE_HTTP_PORT}"
    --http_metadata_server_host=0.0.0.0
    --rpc_port="${MOONCAKE_RPC_PORT}"
    --enable_offload=true
)
if [ "${MOONCAKE_MODE}" = "standalone-store" ]; then
    MOONCAKE_MASTER_ARGS+=(--metrics_port="${MOONCAKE_METRICS_PORT}")
    if [ "${MOONCAKE_OFFLOAD_ON_EVICT}" = "1" ]; then
        MOONCAKE_MASTER_ARGS+=(--offload_on_evict=true)
    fi
else
    MOONCAKE_MASTER_ARGS+=(
        --offload_on_evict=true
        --promotion_on_hit=true
        --promotion_admission_threshold=1
    )
fi
if [ -n "${MOONCAKE_MASTER_LOG}" ]; then
    "${MOONCAKE_MASTER_ARGS[@]}" > "${MOONCAKE_MASTER_LOG}" 2>&1 &
else
    "${MOONCAKE_MASTER_ARGS[@]}" &
fi
MOONCAKE_PID=$!

# Wait for mooncake_master RPC port
echo "[mooncake] Waiting for mooncake_master (pid=${MOONCAKE_PID})..."
for i in $(seq 1 15); do
    sleep 1
    if ! kill -0 "${MOONCAKE_PID}" 2>/dev/null; then
        echo "[mooncake] ERROR: mooncake_master exited early" >&2
        if [ -n "${MOONCAKE_MASTER_LOG}" ]; then
            tail -200 "${MOONCAKE_MASTER_LOG}" >&2 || true
        fi
        exit 1
    fi
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${MOONCAKE_RPC_PORT})); s.close()" 2>/dev/null; then
        echo "[mooncake] mooncake_master ready"
        break
    fi
    if [ "$i" -eq 15 ]; then
        echo "[mooncake] WARNING: RPC port not responding after 15s, continuing anyway"
    fi
done

if [ "${MOONCAKE_MODE}" = "standalone-store" ]; then
    mkdir -p "${MOONCAKE_SSD_DIR}"
    echo "[mooncake] Starting mooncake_client owner..."
    MC_STORE_CLIENT_MIN_PORT="${MOONCAKE_OWNER_SEGMENT_PORT}" \
    MC_STORE_CLIENT_MAX_PORT="${MOONCAKE_OWNER_SEGMENT_PORT}" \
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="${MOONCAKE_SSD_DIR}" \
    MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="${MOONCAKE_SSD_LOCAL_BUFFER_SIZE}" \
    MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="${MOONCAKE_SSD_TOTAL_SIZE_LIMIT}" \
    MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS="${MOONCAKE_SSD_HEARTBEAT_INTERVAL}" \
    MOONCAKE_OWNER_LOG="${MOONCAKE_OWNER_LOG}" \
    bash -c '
        if [ -n "${MOONCAKE_OWNER_LOG}" ]; then
            exec mooncake_client "$@" > "${MOONCAKE_OWNER_LOG}" 2>&1
        fi
        exec mooncake_client "$@"
    ' mooncake_client \
    --host="${MOONCAKE_OWNER_HOST}" \
    --metadata_server=P2PHANDSHAKE \
    --master_server_address="127.0.0.1:${MOONCAKE_RPC_PORT}" \
    --protocol=tcp \
    --global_segment_size="${MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE}" \
    --port="${MOONCAKE_OWNER_RPC_PORT}" \
    --enable_offload=true \
    &
    MOONCAKE_OWNER_PID=$!

    echo "[mooncake] Waiting for mooncake_client owner (pid=${MOONCAKE_OWNER_PID})..."
    for i in $(seq 1 15); do
        sleep 1
        if ! kill -0 "${MOONCAKE_OWNER_PID}" 2>/dev/null; then
            echo "[mooncake] ERROR: mooncake_client owner exited early" >&2
            if [ -n "${MOONCAKE_OWNER_LOG}" ]; then
                tail -200 "${MOONCAKE_OWNER_LOG}" >&2 || true
            fi
            exit 1
        fi
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${MOONCAKE_OWNER_HOST}', ${MOONCAKE_OWNER_RPC_PORT})); s.close()" 2>/dev/null; then
            echo "[mooncake] mooncake_client owner ready"
            break
        fi
        if [ "$i" -eq 15 ]; then
            echo "[mooncake] WARNING: owner RPC port not responding after 15s, continuing anyway"
        fi
    done
fi

ROUTER_CONFIG="pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml" \
TP="$TP" \
bash "${SCRIPT_DIR}/infer_multi.sh" "$MODEL_PATH" "$DATA_PATH" "$AGENT_CONFIG"
