# Mooncake Tier Routing — 验证指南

## 进展状态

| 验证项 | 状态 | 说明 |
|--------|------|------|
| mooncake_master 启动 | ✅ | P2PHANDSHAKE 模式 |
| MooncakeDistributedStore 连接 | ✅ | KVCAwareBalancer 侧 |
| MooncakeStoreConnector（vllm 侧写入） | ✅ | 需要 vllm 0.22.0 |
| ZMQ KV events + tier_store 映射 | ✅ | has_hex=True |
| batch_get_replica_desc 返回有效 descriptor | ✅ | is_memory_replica=True |
| get_tier_prefix_hit_rate 返回非 None | ✅ | CPU tier 命中率 ~0.73 |
| 路由器按 tier 打分路由 | ✅ | 有 cache 的 replica 得 0.80 vs 0.06 |
| SSD tier (is_local_disk_replica=True) | ✅ | standalone-store wrapper E2E 已通过，见下方 2026-06-29 结果 |

### 2026-06-29 SSD E2E 结果

已使用 `scripts/infer_multi_mooncake.sh` 的 `MOONCAKE_MODE=standalone-store` 路径完成 SWE-bench E2E SSD 验证。日志保存在：

```text
/data1/lln/mooncake_ssd_e2e_20260629_122004_forced_tier/
```

关键结论：
- E2E 退出码为 `0`。
- vLLM 进入 `Mooncake mode=standalone-store`，`enable_offload=True`。
- router 使用 `UNI_AGENT_FORCE_TIER_SLOW_PATH=1` 打开验证用 slow path，而不是修改 `load_threshold`。
- `get_tier_prefix_hit_rate: tier='ssd'` 出现非零 `available`，`desc_map_hits` 也为非零。
- master 日志显示 `SSD Storage: 1021.50 MB / 1.00 GB`，`ssd_store` 目录约 `1071335576` bytes。

本次保留旧 CPU tier 复现说明和下面的源码调研结论；SSD 端到端验证应优先使用 standalone-store wrapper 路径。

---

## 环境准备

### 1. Python 包版本要求

```bash
# 检查
pip show vllm torch

# 需要（与默认镜像不同！）
# vllm: 0.22.0（默认镜像是 0.18.0，不含 MooncakeStoreConnector）
# torch: 2.11.0+cu130

# 升级 vllm（按顺序执行）
apt-get install -y cuda-cudart-13-0 cuda-cupti-13-0
pip install torch==2.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.26.0 torchaudio==2.11.0 \
  transformers==5.12.1 xgrammar==0.2.3 mistral_common==1.11.5 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm==0.22.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

# uni-agent 必须以 editable 方式安装（Ray worker 需要能 import）
pip install -e /data1/lln/uni-agent --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 确认 MooncakeStoreConnector 可用

```bash
python3 -c "
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
print('MooncakeStoreConnector:', 'MooncakeStoreConnector' in KVConnectorFactory._registry)
"
# 期望输出: MooncakeStoreConnector: True
```

---

## 快速验证（无需 vllm，只验证查询逻辑）

不依赖 vllm，直接用 Python 向 mooncake store 写入测试数据，验证 `get_tier_prefix_hit_rate` 返回非 None。

```bash
cd /data1/lln/uni-agent
python3 verify_tier_hit_rate.py
```

预期输出：
```
[1] Starting mooncake_master...
  mooncake_master ready (pid=...)
  MooncakeDistributedStore connected
[2] Writing test KV block to mooncake store...
  put(key=...aaaaaaaa) → 0
[3] Verifying via RouteDataProvider...
  local_hash: 13063670922251535405
  remote_hex: aaaaaaaaaaaaaaaa...
  tier_store mappings: 1
  get_tier_prefix_hit_rate(tier='cpu') = 1.0  ✓ PASS
  get_tier_prefix_hit_rate(tier='ssd') = 0.0  ✓ PASS
```

---

## 完整端到端验证（含 vllm + SWE-bench agent）

### 脚本：`scripts/infer_multi_mooncake.sh`

这个脚本做了三件事：
1. 启动 `mooncake_master`（等 RPC 端口 ready）
2. 在 `MOONCAKE_MODE=standalone-store` 时启动持有 CPU pool 和 FileStorage SSD tier 的 `mooncake_client` owner
3. 生成 `MOONCAKE_CONFIG_PATH` JSON 配置文件
4. 调用 `scripts/infer_multi.sh`，附带 `ROUTER_CONFIG` 和 `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`

默认 `MOONCAKE_MODE=standalone-store`；如需复现旧 CPU tier wrapper 行为，可显式设置 `MOONCAKE_MODE=embedded`。

### 最简启动（1个replica，验证基础流程）

```bash
cd /data1/lln/uni-agent

N=1 TP=2 NGPUS=2 NWORKERS=1 MAX_SAMPLES=1 \
PROMPT_LEN=4096 RESPONSE_LEN=4096 \
bash scripts/infer_multi_mooncake.sh /data1/models/Qwen/Qwen3-4B 2>&1 | tee /tmp/test.log
```

观察日志中的关键行：
```
[KVCAwareBalancer] MooncakeTierStore injected into VLLMKVEventCollector
[KVCAwareBalancer] MooncakeDistributedStore connected for tier queries
[VLLMHttpServer] MooncakeStoreConnector initialized, ...
[KVCAwareBalancer] BlockStored: has_token_ids=True, has_hex=True
```

### 触发 tier routing（2个replica，验证路由打分）

**第一步**：设置 `UNI_AGENT_FORCE_TIER_SLOW_PATH=1`，强制验证用 slow path（否则 GPU cache hit 后就走 fast path，不查 tier）：

**第二步**：运行（4卡，TP=2，dp=2，2个replica）：

```bash
UNI_AGENT_FORCE_TIER_SLOW_PATH=1 \
N=2 TP=2 NGPUS=4 NWORKERS=2 MAX_SAMPLES=2 \
PROMPT_LEN=4096 RESPONSE_LEN=4096 \
bash scripts/infer_multi_mooncake.sh /data1/models/Qwen/Qwen3-4B 2>&1 | tee /tmp/test_tier.log
```

**观察目标日志**（大约推理开始后 1-2 分钟出现）：

```
score(): path=slow (forced tier cache)
get_tier_prefix_hit_rate: tier='cpu' total_blocks=565 available=564
get_tier_prefix_hit_rate: keys=1128 desc_map_hits=504
score(): final scores 172.17.0.3:XXXX=0.8081, 172.17.0.3:YYYY=0.0587
```

`available=564` 表示 tier_store 有 564 个映射，`desc_map_hits=504` 表示 mooncake store 返回了 504 个有效 descriptor，高分 replica 被路由。

**第三步**：验证完 unset `UNI_AGENT_FORCE_TIER_SLOW_PATH`。`load_threshold` 保持默认 `0.1`。

### 参数说明

| 参数 | 含义 | 验证建议值 |
|------|------|-----------|
| `N` | 每个 prompt 的 rollout 数 | 2（让同一 prompt 发两次，第二次触发 slow path） |
| `TP` | vllm tensor parallel size | 2 |
| `NGPUS` | 使用 GPU 数，`dp = NGPUS/TP` | 4（2个replica）|
| `NWORKERS` | agent rollout worker 数 | 2 |
| `MAX_SAMPLES` | SWE-bench 样本数 | 2 |
| `PROMPT_LEN` | prompt 长度上限 | 4096（Qwen3-4B 限制 40960） |
| `RESPONSE_LEN` | response 长度上限 | 4096 |

### 常见问题

**vllm 报 `cumem allocator not supported`**
- 原因：vllm 0.22 默认开 sleep_mode，需要 H100+
- 已修复：`parallel_infer.py` 里有 `enable_sleep_mode=False`

**ZMQ `Address already in use`**
- 上一次进程的端口未释放，等几秒或 kill 残留进程：
```bash
pgrep -af "mooncake_master\|parallel_infer" | grep -v defunct | awk '{print $1}' | xargs -r kill
sleep 10
```

**SWE-bench agent 不发请求（`Running: 0 reqs`）**
- Docker 容器启动慢，等 1-2 分钟
- N>1 时多容器并发可能有端口映射问题（Docker-in-container 已知问题），此时用 N=1

---

## SSD Tier 验证（standalone-store wrapper）

当前推荐路径是直接使用 `scripts/infer_multi_mooncake.sh`：

```bash
RUN_DIR=/data1/lln/mooncake_ssd_e2e_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

MOONCAKE_MODE=standalone-store \
MOONCAKE_LOG_DIR="$RUN_DIR" \
MOONCAKE_SSD_DIR="$RUN_DIR/ssd_store" \
UNI_AGENT_FORCE_TIER_SLOW_PATH=1 \
N=2 TP=2 NGPUS=4 NWORKERS=2 MAX_SAMPLES=2 \
PROMPT_LEN=4096 RESPONSE_LEN=4096 \
bash scripts/infer_multi_mooncake.sh /data1/models/Qwen/Qwen3-4B \
2>&1 | tee "$RUN_DIR/e2e.log"
```

观察目标：
- `e2e.log` 出现 `Mooncake mode=standalone-store`。
- `e2e.log` 出现 `score(): path=slow (forced tier cache)`。
- `e2e.log` 出现 `get_tier_prefix_hit_rate: tier='ssd' ... available=` 和非零 `desc_map_hits`。
- `mooncake_master.log` 出现 `SSD Storage` 增长和 `PutEnd` 活动。
- `$RUN_DIR/ssd_store` 有实际文件数据。

`OBJECT_NOT_FOUND` / `REPLICA_IS_NOT_READY` 在批量 tier 查询中可能出现，表示部分 block 查询 miss 或副本尚未 ready；如果同一轮日志里 `desc_map_hits` 非零、SSD `available` 非零且 E2E 退出码为 `0`，不按失败处理。

### 问题描述
以下是旧 wrapper 无法触发 SSD tier 时的调研结论，作为排查参考保留。

### 源码调研结论

1. `scripts/infer_multi_mooncake.sh` 只启动了 `mooncake_master`，没有启动持有
   FileStorage 的 `mooncake_client` owner。
2. wrapper 生成的 `MOONCAKE_CONFIG_PATH` 没有 `mode` 和 `enable_offload` 字段；
   vLLM `MooncakeStoreConfig` 因此默认进入 `embedded` 模式，且
   `enable_offload=False`。
3. vLLM worker 调用 `MooncakeDistributedStore.setup(...)` 时只传
   `metadata_server/global_segment_size/local_buffer_size/protocol/device_name/master`，
   不会把 `enable_offload` 传给 Mooncake Python/C++ setup，也不会在 worker 里创建
   FileStorage。
4. `LOCAL_DISK` descriptor 的来源是 FileStorage owner：`mooncake_client
   --enable_offload=true` 创建 FileStorage，挂载 `LOCAL_DISK` segment，heartbeat 从
   master 拉 offload task，落盘后调用 `NotifyOffloadSuccess`；master 再为对象添加
   `LOCAL_DISK` replica。
5. `--offload_on_evict=true` 会把 offload 从 `PutEnd` 延后到 eviction 路径。默认
   `enable_offload=true` 且不设置 `offload_on_evict` 时，`PutEnd` 会把完成的
   MEMORY replica 直接推入 offload queue，更适合作为第一轮 SSD descriptor 验证。
6. eviction watermark 不是物理机 RAM 使用率。Mooncake master 使用的是
   `mem_allocated_size / mem_total_capacity`，即 Mooncake 已挂载 MEMORY segment 的
   已分配比例。机器有 256GB RAM 不是直接条件；真正的分母是所有已挂载内存段容量。

源码依据：
- `/data1/lln/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
- `/data1/lln/vllm/docs/features/mooncake_store_connector_usage.md`
- `/data1/lln/Mooncake/mooncake-store/src/master_service.cpp`
- `/data1/lln/Mooncake/mooncake-store/src/real_client.cpp`
- `/data1/lln/Mooncake/mooncake-store/src/file_storage.cpp`
- `/data1/lln/Mooncake/mooncake-store/tests/offload_on_evict_test.cpp`

### 旧 wrapper 为什么只能看到 CPU tier

改造前的 wrapper master 参数是：

```bash
mooncake_master \
    --enable_offload=true \
    --offload_on_evict=true \
    --promotion_on_hit=true
```

但没有本地磁盘 owner，也没有 FileStorage heartbeat，所以 master 无法收到
`NotifyOffloadSuccess`，自然不会出现 `is_local_disk_replica=True`。

同时，改造前的 wrapper 写给 vLLM 的配置等价于：

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "master_server_address": "127.0.0.1:50051",
    "global_segment_size": 2147483648,
    "local_buffer_size": 2147483648,
    "protocol": "tcp",
    "device_name": ""
}
```

这会走 vLLM 默认 `mode="embedded"`、`enable_offload=false`，只能稳定验证 CPU tier。当前
`scripts/infer_multi_mooncake.sh` 已增加 `MOONCAKE_MODE=standalone-store` 默认路径，不再受这个限制。

### 手工验证路径 A：standalone-store + PutEnd offload

一般不需要手工执行本段，优先使用上面的 wrapper 命令。本段保留给需要逐步拆开
master、owner、vLLM 三层时排查使用。第一轮只验证 `LOCAL_DISK` descriptor 是否能出现，建议先不要带
`--offload_on_evict=true`，避免把验证点混到 eviction/watermark/lease 逻辑里。

**1. 启动 master**

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=8080 \
  --http_metadata_server_host=0.0.0.0 \
  --rpc_port=50051 \
  --metrics_port=19003 \
  --enable_offload=true
```

`--metrics_port` 如果本机 `9003` 未占用可以不改；这里显式设成 `19003` 是为了避免
默认 admin/metrics 端口冲突。

**2. 启动持有 CPU pool + SSD tier 的 owner**

`MOONCAKE_PREFERRED_SEGMENT` 必须匹配 owner 挂载的 MEMORY segment name。当前
Mooncake `mooncake_client --host=127.0.0.1 --port=12453` 中，`--port` 是 client RPC
服务端口，不是 segment name。segment name 来自 `MC_STORE_CLIENT_MIN_PORT/MAX_PORT`
控制的自动端口，因此推荐显式固定到非 ephemeral 端口（50053 属于 Linux 常见
ephemeral 范围，不能作为该自动端口范围）。

```bash
mkdir -p /tmp/mooncake_ssd

export MC_STORE_CLIENT_MIN_PORT=12353
export MC_STORE_CLIENT_MAX_PORT=12353
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/tmp/mooncake_ssd
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=67108864
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=1073741824
export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1

mooncake_client \
  --host=127.0.0.1 \
  --metadata_server=P2PHANDSHAKE \
  --master_server_address=127.0.0.1:50051 \
  --protocol=tcp \
  --global_segment_size="4 GB" \
  --port=12453 \
  --enable_offload=true
```

owner 日志应包含：

```text
Successfully created client on port 12353
Offload RPC server started on port ...
IsEnableOffloading result: true
```

**3. 给 vLLM 使用 standalone-store 配置**

```bash
cat >/tmp/mooncake_standalone_config.json <<'JSON'
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": true
}
JSON

export MOONCAKE_CONFIG_PATH=/tmp/mooncake_standalone_config.json
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:12353
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
```

vLLM 日志应包含：

```text
Mooncake mode=standalone-store (... preferred_segment=127.0.0.1:12353, enable_offload=True)
```

**4. 用普通脚本跑 E2E**

只有在手工启动 master/owner 并手工导出 `MOONCAKE_CONFIG_PATH` 时，才复用普通脚本：

```bash
ROUTER_CONFIG="pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml" \
N=2 TP=2 NGPUS=4 NWORKERS=2 MAX_SAMPLES=2 \
PROMPT_LEN=4096 RESPONSE_LEN=4096 \
bash scripts/infer_multi.sh /data1/models/Qwen/Qwen3-4B 2>&1 | tee /tmp/test_ssd.log
```

正常 wrapper 验证不需要临时修改 `uni_agent/llm_router/configs/kvc_aware_router.yaml`。

**5. 观察目标**

- owner 日志出现 heartbeat/offload 相关日志。
- `batch_get_replica_desc` 返回的 descriptor 中出现 `is_local_disk_replica=True`。
- `get_tier_prefix_hit_rate(tier='ssd')` 从 `0.0` 变为大于 0 的值。
- 如果刚写入后还是 0，先等待 1-2 个 FileStorage heartbeat 周期再查。

### 推荐验证路径 B：eviction-triggered offload

只有在路径 A 成功后，再验证 `--offload_on_evict=true`。此时要同时满足：

1. master 带 `--enable_offload=true --offload_on_evict=true`。
2. owner FileStorage 已启动并 `IsEnableOffloading result: true`。
3. 对象 lease 过期或可被 eviction 选中。
4. `mem_allocated_size / mem_total_capacity > eviction_high_watermark_ratio`，或
   `need_mem_eviction_` 被置位。

要降低触发难度，应减少 owner 的 `--global_segment_size` 和各 requester 的
`global_segment_size`，而不是按物理机 RAM 估算。没有 FileStorage owner 时，缩小内存段
只会更快遇到内存段写满/无可用 handle，不会产生 `LOCAL_DISK` descriptor。

### `_descriptor_matches_tier` 代码路径已就绪
```python
# uni_agent/llm_router/collectors/provider.py
def _descriptor_matches_tier(desc, tier):
    if tier == "cpu":
        return desc.is_memory_replica()         # ✅ 已验证
    if tier == "ssd":
        for method in ("is_local_disk_replica", "is_nof_replica"):
            m = getattr(desc, method, None)
            if callable(m) and m():
                return True                     # ⚠️ 代码正确，未能触发场景
    return False
```
