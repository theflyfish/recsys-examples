# KVCacheManager 快速参考卡片

> 这是一份速查表，配合深度分析文档使用。

---

## 概念速查

| 概念 | 定义 | 典型值 |
|------|------|--------|
| **User ID** | 推荐系统用户唯一标识，KV 缓存的查找键 | int64，1-1000M |
| **Sequence** | 一个用户的行为历史（item + action 交替） | 10-512 tokens |
| **Page** | GPU 缓存的最小分配单位 | 64 tokens/page |
| **Paged Cache** | GPU 内存按页管理的 KV 存储 | [num_pages, page_size, num_heads, head_dim] |
| **Metadata** | 分配、索引、同步信息（位置敏感，一块 GPU buffer） | 5B + 4 + 2T int32s |
| **LRU** | 最近最少使用驱逐策略 | 按 GPU 内最后访问时间 |
| **Onboard** | 主机→GPU 异步拷贝（H2D） | cudaMemcpyAsync + event |
| **Offload** | GPU→主机异步拷贝（D2H） | 后台进行，与 post/MLP 重叠 |
| **Stream Wait** | 在计算 stream 上等待 H2D event（按层） | 允许第 L 层计算与第 L-1 层 H2D 重叠 |

---

## Python API 速查

### 初始化

```python
from recsys_kvcache_manager import KVCacheManager, KVCacheConfig

config = KVCacheConfig(
    num_layers=12, num_heads=32, head_dim=96,
    page_size=64,                              # tokens/page
    offload_chunksize=256,                     # offload 基本单位
    num_primary_cache_pages=4096,              # GPU 总页数
    num_buffer_pages=256,                      # 保留给临时页
    host_capacity_per_layer=1024*1024*1024,   # 1GB per layer
    max_batch_size=32, max_seq_len=512,
    dtype=torch.bfloat16, device=0,
    host_kvstorage_backend="native")           # or "flexkv"

kvcache_mgr = KVCacheManager.from_config(config)
```

### 推理流程

```python
# 1. 查询
index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, seq_lengths)
#   ↳ lookup_res.cached_lengths: [B] 已缓存长度

# 2. 分配
kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
#   ↳ kvcache_metadata.kv_cache_table: GPU 分页表指针
#   ↳ kvcache_metadata.kv_indices: 分配的页 ID
#   ↳ kvcache_metadata.kv_indptr: CSR 指针

# 3. 异步上板
kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)

# 4. 计算（embedding/HSTU）
...

# 5. HSTU 内部（每层）
#   - append_kvcache：写新 KV
#   - stream_wait_layer(layer_idx)：按层等待 H2D
#   - attention：从分页表读取

# 6. 异步下板（post 前）
kvcache_mgr.offload_try_wait()      # 轮询之前的下板任务
kvcache_mgr.offload_launch(index_meta, kvcache_metadata)

# 7. 后处理
...
```

---

## 内存布局速查

### GPU Tensor

```
shape: [num_layers=12, num_pages=4096, 2(k/v), page_size=64, num_heads=32, head_dim=96]
size:  12 * 4096 * 2 * 64 * 32 * 96 * 2 bytes ≈ 37.8 GB
```

### Metadata Buffer

```
int32 buffer，长度 = 5*B + 4 + 2*T（其中 B=batch_size, T=num_new_tokens）

布局（偏移）：
 0 : B+1                      → kv_indptr        [CSR 指针]
 B+1 : 2B+1                   → kv_last_page_len [最后页长度]
 2B+1 : 3B+1                  → total_history_lengths
 3B+1 : 4B+2                  → total_history_offsets
 4B+2 : 4B+3                  → new_history_nnz_cuda [GPU scalar]
 4B+3 : 5B+4                  → new_history_offsets
 5B+4 : 5B+4+T                → batch_indices    [token→序列映射]
 5B+4+T : ...                 → position         [token→位置映射]
```

---

## 关键数据结构

### KVLookupResult

```python
@dataclass
class KVLookupResult:
    user_ids: Tensor                    # [B]
    gpu_cached_start_indices: Tensor    # [B]
    gpu_cached_lengths: Tensor          # [B]
    host_cached_start_indices: Tensor   # [B]，恒为 0
    host_cached_lengths: Tensor         # [B]
    cached_start_indices: Tensor        # [B]，合并后
    cached_lengths: Tensor              # [B]，合并后
```

**合并规则**：
```
if host_len == 0: → 用 GPU 结果
elif gpu_len == 0: → 用 host 结果
else: → max(host_len, gpu_start+gpu_len)，要求无空洞
```

### KVCacheMetadata

```python
@dataclass
class KVCacheMetadata:
    page_ids_gpu_buffer: Tensor         # [num_pages]，页 ID
    metadata_gpu_buffer: Tensor         # 上述扁平 buffer
    
    # CSR 风格索引
    kv_indices: Tensor                  # 页 ID [num_pages]
    kv_indptr: Tensor                   # 页范围指针 [B+1]
    kv_last_page_len: Tensor            # [B]
    
    # append/attention 参数
    batch_indices: Tensor               # [num_new_tokens]，token→seq
    position: Tensor                    # [num_new_tokens]，token→pos
    new_history_nnz: int                # 新 token 总数
    
    # Attention 读取长度
    kv_seqlens: Tensor                  # [B]，history + candidates
    kv_seqlen_offsets: Tensor           # [B+1]
    
    # 异步句柄
    kv_onload_handle: HostKVTaskHandle  # H2D 句柄，支持 stream_wait_layer
    
    # 分页表指针
    kv_cache_table: List[Tensor]        # [num_layers][pages][2][...] 
```

---

## 命令行/环境变量

```bash
# FlexKV 后端配置
export FLEXKV_GPU_REGISTER_PORT="ipc:///tmp/flexkv_server_gpu_register"

# 调试模式
export DEBUG_KVCACHE=1
export DUMP_KVCACHE=1  # 转储元数据
```

---

## 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `allocate_kvcache` 变慢 | GPU 页不足，LRU 驱逐 | 增加 `num_primary_cache_pages` |
| Onboard 超时 | H2D 带宽饱和 | 增加 `onload_timeout_ms`；分散 batch |
| Offload 失败且策略 fail_close | 主机缓存满或 I/O 错误 | 增加 `host_capacity_per_layer`；检查磁盘 |
| 内存 OOM（GPU） | 总缓存超过显存 | 降低 `num_primary_cache_pages` 或减少 batch |
| CUDA Graph 不匹配 | 批大小/序列长度超出预设范围 | 检查 `cudagraph_configs` 中的范围 |
| 注意力结果不对 | `kv_last_page_len` 计算错误 | 检查 append_kvcache 是否正确更新 |

---

## 性能调优检查表

- [ ] GPU 缓存命中率 > 80%（用 `cache_hit_rate` 指标）
- [ ] H2D 与 embedding 重叠 > 90%（用 timeline trace）
- [ ] Offload 与 post/MLP 重叠 > 80%
- [ ] LRU 驱逐频率 < 每 100 batch 驱逐 1 次
- [ ] GPU 缓存利用率 > 70%（实际用页 / 总页数）
- [ ] Metadata buffer 切片无越界（运行 debug build 验证）

---

## 关键文件导航

| 文件 | 职责 |
|------|------|
| `kvcache_manager.py` | 上层编排（lookup → allocate → onboard → offload） |
| `gpu_kvcache_manager.py` | GPU 分页表管理 + LRU 驱逐 |
| `native_host_kvcache_manager.py` | Host 存储（H2D/D2H + 按层等待） |
| `kvcache_metadata.py` | 元数据 buffer 布局与创建 |
| `kvcache_utils.py` | KVLookupResult 合并逻辑 |
| `paged_hstu_infer_layer.py` | `append_kvcache` 调用 + attention kernel 接口 |
| `hstu_block_inference.py` | CUDA Graph capture / replay（分层） |

---

## 测试要点

```python
# 1. 基本功能测试
def test_lookup_allocate():
    """验证查询与分配的正确性"""
    user_ids = torch.tensor([1, 2, 3])
    seq_lengths = torch.tensor([100, 200, 150])
    # lookup → allocate → 验证 kv_indptr/kv_last_page_len

# 2. 异步同步测试
def test_onboard_offload():
    """验证 H2D/D2H 的可靠性与时序"""
    # onboard_launch + offload_launch + 轮询完成

# 3. LRU 驱逐测试
def test_lru_eviction():
    """验证 LRU 正确驱逐最老用户"""
    # 填满 GPU 页 → 新 batch 应驱逐最老

# 4. CUDA Graph 一致性
def test_cudagraph_correctness():
    """验证 Graph 回放结果与 eager 相同"""
    # 同批数据，eager vs graph 对比
```

---

## 文献与扩展阅读

1. **原始设计文档**：`README.md`（架构总览）
2. **代码级深度分析**：`KVCACHE_CODE_FLOW_ANALYSIS.md`
3. **企业级技术指南**：`KVCACHE_INFRA_EXPERT_GUIDE.md`（本文档配套）
4. **论文参考**：FlexKV（http://github.com/taco-project/FlexKV）
5. **CUDA 优化**：NVIDIA CUDA Best Practices Guide（stream 管理、kernel launch）
