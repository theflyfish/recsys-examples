# KVCacheManager 模块企业级深度分析

> **目标读者**：已掌握 CUDA/C++ 与 PyTorch 的系统工程师。  
> **完成阅读后预期**：能独立设计/修改/优化 KVCache 系统，理解每个设计决策的 tradeoff。

---

## Part I：架构设计理念与整体思路

### 问题陈述

**背景**：生成式推荐模型在推理时需要重用用户的行为序列 KV。不同于 LLM（不同用户的 prompt 可能有相同前缀），推荐系统中：
- 不同用户的行为序列几乎 **无公共前缀**
- 重用粒度是 **per-user**，而非 per-token-sequence
- 历史可达 **数百个 token**，逐次request 增量很小（通常 <10 个新 token）

**传统 LLM KVCache 方案的问题**：
1. token 序列 hash/trie 比对开销大（O(n)）
2. 多用户并发时内存碎片严重
3. 无法有效利用分层存储（GPU→Host→SSD）

**RecSys KVCacheManager 的解方**：
- **User-ID based lookup**：O(1) 查询，直接 user_id → 缓存指针
- **Paged GPU cache**：GPU 内存按固定页（page）管理，配 LRU 驱逐，支持 HSTU attention kernel 直接读取
- **Hierarchical storage**：GPU full-speed → Host bandwidth-limited → SSD/remote capacity-limited，异步重叠 H2D/D2H
- **Async-first design**：onboard/offload 与计算重叠，按层 stream 同步允许层间计算与上层 H2D 重叠

### 核心设计决策

| 决策 | 理由 | 代价 |
|------|------|------|
| **User-ID hash table lookup** | 推荐系统天然的分片键，O(1)，无碎片 | 不能共享不同用户的前缀 |
| **Paged storage** | 支持 HSTU kernel 直接索引，避免 gather; 页粒度驱逐比 token 粒度更高效 | 页内碎片（最后一页可能未满） |
| **Async H2D/D2H** | 与 embedding/HSTU/post 计算重叠，降低端到端延迟 | 需复杂的 stream 管理与错误恢复 |
| **Per-layer onboard wait** | 第 L 层计算与第 L-1 层 H2D 重叠 | 仅 native 后端支持；需 layer-aware handle |
| **Fail-open 容错** | H2D/D2H 失败不中止推理，保证可用性 | 可能返回陈旧数据（但如果 offload 失败，GPU 数据仍在） |

---

## Part II：核心数据结构与内存布局

### 2.1 GPU KVCache 张量布局

**代码**：`gpu_kvcache_manager.py:52-64`

```python
gpu_kvcache_tensor: [num_layers, num_primary_cache_pages, 2, page_size, num_heads, head_dim]
```

**内存布局图**：

```
GPU VRAM 视图（以 bfloat16 为例）
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 0 (num_primary_cache_pages, 2, page_size, num_heads, head_dim)       │
│  ├─ Page 0     ├─ K [page_size, num_heads, head_dim]                        │
│  │             └─ V [page_size, num_heads, head_dim]                        │
│  ├─ Page 1     ├─ K [page_size, num_heads, head_dim]                        │
│  │             └─ V [page_size, num_heads, head_dim]                        │
│  └─ ...                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 1 ...                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ ...                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

单页大小（一个 [K,V] pair）：
  2 * page_size * num_heads * head_dim * sizeof(bfloat16)
  = 2 * 64 * 32 * 96 * 2 bytes  = 768 KB (典型值)

GPU 总占用（典型）：
  12 layers * 4096 pages * 768 KB ≈ 37.8 GB
```

**为什么这个布局**？
1. **层为最外层**：注意力逐层计算，缓存局部性好（每层用一个子张量）；
2. **页为次层**：便于 CSR 风格的索引（`kv_indptr` 指向页范围）；
3. **K/V 并排**：attention kernel 需同时访问 K 和 V，内存连续性优化；
4. **token×head×dim 作最内**：在 token 维度连续，便于向量化读写。

### 2.2 KVCacheMetadata 扁平内存模型

**关键洞察**：为了兼容 CUDA Graph，所有 metadata 不用对象指针，而是一块 GPU 内存 buffer 切片。

**代码**：`kvcache_metadata.py:62-130`

```python
metadata_gpu_buffer: torch.Tensor  # 一块扁平 int32 buffer
  形状 [5*batch_size + 4 + 2*num_new_tokens]
  
切片布局（偏移量在下面）：
  [0 : batch_size+1]                        → kv_indptr         # CSR 页索引指针
  [batch_size+1 : 2*batch_size+1]           → kv_last_page_len  # 最后页长度
  [2*batch_size+1 : 3*batch_size+1]         → total_history_lengths
  [3*batch_size+1 : 4*batch_size+2]         → total_history_offsets
  [4*batch_size+2 : 4*batch_size+3]         → new_history_nnz_cuda  # GPU scalar
  [4*batch_size+3 : 5*batch_size+4]         → new_history_offsets
  [5*batch_size+4 : 5*batch_size+4+num_new_tokens]              → batch_indices
  [5*batch_size+4+num_new_tokens : ...]      → position
```

**为什么扁平化**？
1. CUDA Graph capture 需要静态内存指针——对象指针会变，但 buffer 视图不会；
2. 单次内存分配，减少碎片；
3. GPU→C++ 端只需传指针偏移，无需拷贝元数据。

**内存布局示例**（batch_size=2, num_new_tokens=3）：

```
metadata_gpu_buffer 的实际值示例：
Index    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14  ...
Value [page0 p0   pl0  h0   h0o  h0o  n0   new  0    2    1    1    0    0    0  ...
        ptr] len  len  len  set  ofs  nz   offs                    pos0 pos1 pos2 ...]
        
kv_indptr        = [0, 2, 3]       # seq0用pages[0:2], seq1用pages[2:3]
kv_last_page_len = [32, 16]        # seq0最后页32个有效token，seq1最后页16个
...
```

### 2.3 `KVLookupResult` 的语义与合并规则

**设计**：记录「当前查询 batch 的每个用户在 GPU+Host 两级缓存中的覆盖范围」。

**代码**：`kvcache_utils.py:28-45`

```python
@dataclass
class KVLookupResult:
    user_ids: torch.Tensor
    
    # GPU 侧
    gpu_cached_start_indices: torch.Tensor  # [B]，每用户 GPU 缓存起始 token idx
    gpu_cached_lengths: torch.Tensor        # [B]，每用户 GPU 缓存长度
    
    # Host 侧
    host_cached_start_indices: torch.Tensor # [B]，恒为 0（host 从序列开头缓存）
    host_cached_lengths: torch.Tensor       # [B]，每用户 host 缓存长度
    
    # 合并后
    cached_start_indices: torch.Tensor      # [B]，最终「已缓存」起始
    cached_lengths: torch.Tensor            # [B]，最终「已缓存」长度
```

**合并规则**（`kvcache_utils.py:46-124`），逐样本 i 计算：

```python
if host_cached_lengths[i] == 0:
    # Host 无缓存 → 返回 GPU 命中
    cached_start_indices[i] = gpu_cached_start_indices[i]
    cached_lengths[i] = gpu_cached_lengths[i]
    
elif gpu_cached_lengths[i] == 0:
    # GPU 无缓存，Host 有 → 返回 Host 命中
    # (要求 host 从 0 缓存)
    assert host_cached_start_indices[i] == 0
    cached_start_indices[i] = 0
    cached_lengths[i] = host_cached_lengths[i]
    
else:
    # 两者都有 → 合并（要求无空洞）
    assert gpu_cached_start_indices[i] <= host_cached_lengths[i]  # 无空洞
    cached_len = max(host_cached_lengths[i], 
                     gpu_cached_start_indices[i] + gpu_cached_lengths[i])
    cached_start_indices[i] = gpu_cached_start_indices[i] or 0
    cached_lengths[i] = cached_len
```

**无空洞假设的含义**：

```
合法配置：
  Host:  [====|====|====]  (0-12 缓存)
  GPU:         [===|===]   (8-16 缓存)
  →merged:    [====|====|===]  (0-16)

非法配置（会 assert fail）：
  Host:  [===|===|===]     (0-9 缓存)
  GPU:              [==]    (12-14 缓存)  ← 10-11 有空洞，不允许！
```

原因：HSTU 注意力需要连续读取，不能跳过中间 token。

---

## Part III：Python 接口详解与 C++ 契约

### 3.1 `KVCacheManager.__init__` 与初始化链

**代码**：`kvcache_manager.py:32-64`

```python
def __init__(self, gpu_kvcache_manager, host_kvstorage_manager, 
             offload_mode="lazy", host_kvstorage_fail_policy="fail_open"):
    self.gpu_kvcache_mgr = gpu_kvcache_manager
    self.host_kvstorage_manager = host_kvstorage_manager
    
    # 关键：把 GPU 缓存表注册给 Host，便于 host 端在 offload 时访问（直接 memcpy）
    self.host_kvstorage_manager.register_gpu_cache_tables(
        self.gpu_kvcache_mgr.get_cache_tables())
    
    self.offload_mode = KVCacheOffloadMode(offload_mode)  # LAZY or EAGER
    self.host_kvstorage_fail_policy = host_kvstorage_fail_policy
    
    self.ongoing_onboard_tasks: List[HostKVTaskHandle] = []
    self.ongoing_offload_tasks: List[HostKVTaskHandle] = []
```

**契约**：
- GPU manager 与 Host manager 是**独立的**，通过 KVCacheManager 协调；
- Host manager 需持有 GPU 缓存表指针，用于 offload 时直接内存拷贝（避免 cudaMemcpy 开销）；
- 任务队列由 manager 维护，上层只需非阻塞轮询（`offload_try_wait`）。

### 3.2 GPUKVCacheManager 深度解析

#### 3.2.1 构造与张量初始化

**代码**：`gpu_kvcache_manager.py:26-81`

```python
class GPUKVCacheManager:
    def __init__(self, num_layers, num_heads, head_dim, num_tokens_per_page,
                 num_tokens_per_chunk, num_primary_cache_pages, num_buffer_pages,
                 max_batch_size, max_sequence_length, dtype, device_idx):
        
        # 1. 参数保存
        self.num_layers = num_layers
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk  # offload 粒度
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_buffer_pages = num_buffer_pages  # 保留给临时页的数量
        
        # 2. GPU 张量分配（一块大的连续内存）
        self.gpu_kvcache_tensor = torch.empty([
            num_layers,
            num_primary_cache_pages,
            2,  # K, V
            page_size,
            num_heads,
            head_dim
        ], dtype=dtype, device=device_idx)
        
        # 3. 按层切分视图（无内存拷贝，只是 reshape）
        self.gpu_kvcache_tables = list(self.gpu_kvcache_tensor.unbind(dim=0))
        # 现在 gpu_kvcache_tables[l] 形状为 [num_primary_cache_pages, 2, page_size, num_heads, head_dim]
        
        # 4. 初始化 C++ 实现
        self.impl_ = GPUKVCacheManagerImpl(
            num_layers, num_heads, head_dim, page_size, chunk_size,
            num_primary_cache_pages, num_buffer_pages, max_batch_size,
            max_sequence_length, device_idx)
        
        # 5. GPU 硬件信息（优化 kernel 调用）
        self.num_sms = torch.cuda.get_device_properties(device_idx).multi_processor_count
```

**性能考量**：
- `torch.empty` 一次分配全部内存，避免多次 malloc（碎片）；
- `unbind(dim=0)` 无内存拷贝，只是改变 shape/stride；
- 记录 SM 数量用于 kernel 参数化（block 粒度）。

#### 3.2.2 查询 `lookup(user_ids)` 的 C++ 契约

**代码**：`gpu_kvcache_manager.py:89-95`

```python
def lookup(self, uids: torch.Tensor) -> KVLookupResult:
    # C++ 实现：维护一个全局 user_id → (start_idx, cached_len) 的哈希表
    cached_start_indices, cached_lengths = self.impl_.lookup(uids)
    return KVLookupResult(
        user_ids=uids,
        gpu_cached_start_indices=cached_start_indices,  # GPU 缓存起始位置
        gpu_cached_lengths=cached_lengths,               # GPU 缓存长度
    )
```

**C++ 实现逻辑**（推断）：
```cpp
// cpp_impl.cu
std::vector<int> gpu_cache_start_indices;   // user_id → GPU 缓存起始位置
std::vector<int> gpu_cache_lengths;         // user_id → GPU 缓存长度

void lookup(torch::Tensor uids, torch::Tensor &out_starts, torch::Tensor &out_lens) {
    int B = uids.size(0);
    int* uid_data = uids.data_ptr<int>();
    int* out_starts_data = out_starts.data_ptr<int>();
    int* out_lens_data = out_lens.data_ptr<int>();
    
    for (int i = 0; i < B; ++i) {
        int uid = uid_data[i];
        auto it = uid_to_cache_info.find(uid);
        if (it != uid_to_cache_info.end()) {
            out_starts_data[i] = it->second.start_idx;
            out_lens_data[i] = it->second.length;
        } else {
            out_starts_data[i] = 0;
            out_lens_data[i] = 0;  // 未缓存
        }
    }
}
```

**性能注意**：
- 查询是 O(B)（batch size），每个 user 的哈希表查询是 O(1) 均摊；
- 数据在 CPU 侧维护（not GPU），查询时 transfer 结果回 GPU（latency ~1-5 us）。

#### 3.2.3 分配 `allocate(uids, seq_hist_lengths, lookup_results, ...)` 的关键

**代码**：`gpu_kvcache_manager.py:97-126`

```python
def allocate(self, uids, seq_hist_lengths, lookup_results, 
             output_kvcache_metadata):
    # 1. 计算新增 token（真正需要新算的）
    new_hist_lengths = seq_hist_lengths - lookup_results.cached_lengths
    # seq_hist_lengths: [B]，该用户完整历史长度（包括已缓存）
    # lookup_results.cached_lengths: [B]，已缓存长度
    # new_hist_lengths: [B]，待新增的 token
    
    # 2. 计算需要多少页
    num_new_tokens = torch.sum(new_hist_lengths).item()
    num_total_pages = torch.sum(
        torch.ceil(seq_hist_lengths.float() / self.page_size).to(torch.int32)
    ).item()
    # 这是该 batch 用户需要的总页数（包括已缓存的）
    
    # 3. 若元数据 buffer 未提供，创建新的
    if output_kvcache_metadata is None:
        output_kvcache_metadata = get_kvcache_metadata_buffer(
            batch_size=uids.size(0),
            num_new_tokens=num_new_tokens,
            num_pages=num_total_pages)
    
    # 4. 将 GPU 缓存表指针嵌入元数据（注意力层会用这个指针读写）
    output_kvcache_metadata.kv_cache_table = self.gpu_kvcache_tables
    
    # 5. **核心**：调用 C++ 实现进行分页分配 + LRU 驱逐
    self.impl_.allocate(
        uids,                                    # [B] 用户 ID
        seq_hist_lengths,                        # [B] 完整历史长度
        lookup_results.host_cached_lengths,     # [B] host 已缓存长度
        output_kvcache_metadata.page_ids_gpu_buffer,    # 输出：分配的页 ID
        output_kvcache_metadata.metadata_gpu_buffer)    # 输出：管理元数据
    
    return output_kvcache_metadata
```

**C++ allocate 的核心逻辑**（伪码）：

```cpp
// cpp_impl.cu
struct PageAllocation {
    std::vector<int> page_ids;      // 该 user 分配的页 ID 列表
    int start_page_idx;             // 在全局 page_ids buffer 中的起始偏移
};

std::unordered_map<int, PageAllocation> user_to_pages;  // user_id → 页分配

void allocate(torch::Tensor uids, torch::Tensor seq_lens, torch::Tensor host_cached_lens,
              torch::Tensor out_page_ids, torch::Tensor out_metadata) {
    
    std::vector<int> free_pages;    // 当前可用页列表
    int total_pages_needed = 0;
    
    // 1. 计算每个 user 需要多少页
    for (int i = 0; i < B; ++i) {
        int uid = uids[i];
        int num_hist_pages = ceil(seq_lens[i] / page_size);
        total_pages_needed += num_hist_pages;
    }
    
    // 2. 若可用页不足，触发 LRU 驱逐
    int available = num_primary_cache_pages - current_used_pages;
    if (total_pages_needed > available) {
        int need_to_evict = total_pages_needed - available;
        std::vector<int> lru_users = find_lru_users(need_to_evict);  // 按最后访问时间排序
        
        for (int evict_uid : lru_users) {
            // 移除该用户的页分配
            auto pages = user_to_pages[evict_uid];
            for (int page_id : pages.page_ids) {
                free_pages.push_back(page_id);
                current_used_pages--;
            }
            user_to_pages.erase(evict_uid);
            // 注意：当前实现**不做 offload**（README 已说明）
        }
    }
    
    // 3. 分配新页给请求的用户
    int out_offset = 0;
    for (int i = 0; i < B; ++i) {
        int uid = uids[i];
        int num_pages = ceil(seq_lens[i] / page_size);
        std::vector<int> allocated_pages;
        
        for (int j = 0; j < num_pages; ++j) {
            if (free_pages.empty()) {
                // 不应该到这里（前面已驱逐了）
                throw std::runtime_error("Not enough pages!");
            }
            int page_id = free_pages.back();
            free_pages.pop_back();
            allocated_pages.push_back(page_id);
            out_page_ids[out_offset++] = page_id;  // 写到 buffer
        }
        
        user_to_pages[uid] = {allocated_pages, out_offset - num_pages};
        update_lru_timestamp(uid);  // 标记该 user 最新访问
    }
    
    // 4. 填充元数据 buffer（kv_indptr, kv_last_page_len 等）
    fill_metadata_buffer(out_metadata, user_to_pages, seq_lens, host_cached_lens);
}
```

**关键点**：
1. **LRU 驱逐**：按最后访问时间戳驱逐最旧的用户（HSTU 推理中，本 batch 用户会更新时间戳）；
2. **不做 offload**：驱逐时直接丢弃页，不转移到 host（这是设计简化，生产系统应改进）；
3. **页 ID 映射**：分配的页 ID 列表写到 `page_ids_gpu_buffer`，供注意力层索引；
4. **元数据填充**：计算 `kv_indptr`（CSR 指针）、`kv_last_page_len`（页内有效长度）等。

#### 3.2.4 offload 准备 `acquire_offload_pages`

**代码**：`gpu_kvcache_manager.py:203-209`

```python
def acquire_offload_pages(self, uids, offloaded_lengths, always_offload=False):
    # 返回可以 offload 的用户列表与页信息（同时锁定这些页）
    return self.impl_.acquire_offload_pages(uids, offloaded_lengths, always_offload)
    # 返回 (offload_user_ids, offload_start_indices, offload_page_indices_list)
```

**C++ 逻辑**（伪码）：

```cpp
// 返回的三元组用于后续 offload_launch
std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
acquire_offload_pages(torch::Tensor uids, torch::Tensor offloaded_lens, bool always_offload) {
    std::vector<int> offload_uids;
    std::vector<int> offload_starts;
    std::vector<std::vector<int>> offload_pages_list;
    
    // LRU 策略选出待驱逐用户（当前 batch 用户除外）
    std::vector<int> candidates = find_candidates_for_offload(uids);
    
    for (int cand_uid : candidates) {
        auto alloc = user_to_pages[cand_uid];
        offload_uids.push_back(cand_uid);
        offload_starts.push_back(offloaded_lens[cand_uid]);  // 该用户在 host 已存的长度
        offload_pages_list.push_back(alloc.page_ids);
        
        // 标记这些页为「locked」，防止并发重用
        for (int page_id : alloc.page_ids) {
            page_lock_flags[page_id] = true;
        }
    }
    
    return {offload_uids, offload_starts, offload_pages_list};
}
```

### 3.3 Host 端存储接口与 Native 后端

#### 3.3.1 `NativeHostKVCacheManager` 架构

**代码**：`native_host_kvcache_manager.py:32-75`

```python
class NativeHostKVCacheManager(HostKVStorageManagerBase):
    def __init__(self, num_layers, num_heads, head_dim, num_tokens_per_page,
                 num_tokens_per_chunk, bytes_capacity_per_layer, max_batch_size,
                 max_sequence_length, onload_timeout_ms=0, offload_timeout_ms=0,
                 dtype=torch.bfloat16, device_idx=0):
        
        self.backend_name = "native"
        self.num_layers = num_layers
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.bytes_capacity_per_layer = bytes_capacity_per_layer
        
        # C++ host 存储实现
        self.impl_ = HostKVStorageImpl(
            num_layers, num_heads, head_dim, page_size, chunk_size,
            bytes_capacity_per_layer, max_batch_size, max_sequence_length,
            device_idx)
        
        self._onload_timeout_ms = onload_timeout_ms
        self._offload_timeout_ms = offload_timeout_ms
```

**Host 端的存储结构**（C++ 侧，伪码）：

```cpp
// 每层一份 host 存储
struct HostKVStorage {
    std::unordered_map<int, std::vector<float16>> kv_data;  // user_id → KV 数据
    std::unordered_map<int, int> user_cached_lengths;       // user_id → 缓存长度
    std::mutex mutex;
};

std::vector<HostKVStorage> host_storage_per_layer;  // [num_layers]

int bytes_per_user_per_layer = ???  // 动态计算
```

#### 3.3.2 查询 `lookup_kvcache(index_meta)` 的语义

**代码**：`native_host_kvcache_manager.py:88-95`

```python
def lookup_kvcache(self, index_meta):
    # C++ 查询：对每个 user，返回 host 中已缓存的长度
    cached_lengths = self.impl_.lookup(index_meta.user_ids)  # [B]
    
    return KVLookupResult(
        user_ids=index_meta.user_ids,
        host_cached_start_indices=torch.zeros_like(cached_lengths),  # 恒为 0
        host_cached_lengths=cached_lengths,                           # 实际缓存长度
    )
```

**C++ lookup**（伪码）：

```cpp
torch::Tensor lookup(torch::Tensor uids) {
    int B = uids.size(0);
    auto result = torch::zeros(B, torch::kInt32);
    int* result_data = result.data_ptr<int>();
    int* uid_data = uids.data_ptr<int>();
    
    for (int i = 0; i < B; ++i) {
        int uid = uid_data[i];
        std::lock_guard<std::mutex> lock(host_storage_per_layer[0].mutex);  // 代表性
        auto it = host_storage_per_layer[0].user_cached_lengths.find(uid);
        result_data[i] = (it != end) ? it->second : 0;
    }
    
    return result;
}
```

**Native host 的限制**（来自 `README.md:89-91`）：
- 至多 1 个 GPU 管理器 + 1 个推理实例
- 需配合 user_id 路由隔离（不同用户分发到不同实例）
- 原因：host 侧全局数据结构无分布式支持

---

## Part IV：异步数据传输机制

### 4.1 Onboard (Host→GPU) 的生命周期

**代码流程**：`kvcache_manager.py:93-108` → `native_host_kvcache_manager.py:97-150`

#### 4.1.1 发起 `onboard_launch`

```python
def onboard_launch(self, index_meta, lookup_result, kvcache_metadata):
    task_handle = self.host_kvstorage_manager.onboard_kvcache_launch(
        index_meta, lookup_result, kvcache_metadata)
    kvcache_metadata.kv_onload_handle = task_handle  # 保存句柄用于后续等待
    return task_handle
```

**Native 实现**（`native_host_kvcache_manager.py:97-150`）：

```python
def onboard_kvcache_launch(self, index_meta, lookup_result, kvcache_metadata):
    # 1. 确定需要从 host 加载哪些数据
    g_end_idxs = lookup_result.gpu_cached_start_indices + lookup_result.gpu_cached_lengths
    h_longer = g_end_idxs < lookup_result.host_cached_lengths
    
    # GPU 缺失而 host 有的范围
    onload_start_indices = torch.where(
        torch.logical_and(lookup_result.gpu_cached_start_indices == 0, h_longer),
        g_end_idxs,
        0)
    onload_lengths = torch.where(
        h_longer,
        lookup_result.host_cached_lengths,
        lookup_result.gpu_cached_start_indices)
    
    # 2. 根据 onload 范围，查出对应的 GPU 页 ID
    onload_paged_ids_list = [
        kvcache_metadata.kv_indices[
            kvcache_metadata.kv_indptr[seq_idx] + onload_start_indices[seq_idx] // self.page_size
            : kvcache_metadata.kv_indptr[seq_idx] + (onload_start_indices[seq_idx] + onload_lengths[seq_idx] + self.page_size - 1) // self.page_size
        ]
        for seq_idx in range(index_meta.user_ids.size(0))
    ]
    
    if torch.sum(onload_lengths).item() == 0:
        # 无数据需搬，直接返回 SKIPPED
        return HostKVTaskHandle(backend="native", status=HostKVTaskStatus.SKIPPED)
    
    # 3. 调用 C++ 发起异步 H2D
    native_handle = KVOnloadHandle(self.num_layers)  # C++ 对象
    self.impl_.onload_kvcache(
        index_meta.user_ids,
        onload_paged_ids_list,
        native_handle)
    
    # 4. 返回句柄，包含必要的元数据供后续恢复
    return HostKVTaskHandle(
        backend="native",
        user_ids=index_meta.user_ids,
        handle=native_handle,
        status=HostKVTaskStatus.LAUNCHED,
        is_layerwise=True,  # native 支持按层等待
        metadata={
            "onboard_start_indices": onload_start_indices,
            "onboard_lengths": onload_lengths})
```

**C++ onload_kvcache**（伪码，核心逻辑）：

```cpp
// kvcache_cpp/native_host_kvcache.cu
class KVOnloadHandle {
public:
    cudaEvent_t layer_ready_events[MAX_LAYERS];  // 每层一个事件，表示该层 H2D 完成
    cudaStream_t h2d_stream;                      // 所有 H2D 在这个 stream 上
};

void onload_kvcache(torch::Tensor uids, 
                    std::vector<torch::Tensor> onload_paged_ids_list,
                    KVOnloadHandle& handle) {
    
    // 为每一层分别发起 H2D
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        
        // 从 host 读出该层的数据
        std::vector<void*> host_ptrs;
        std::vector<int> sizes;
        
        for (int seq_idx = 0; seq_idx < B; ++seq_idx) {
            int uid = uids[seq_idx];
            int* page_ids = onload_paged_ids_list[seq_idx].data_ptr<int>();
            int num_pages = onload_paged_ids_list[seq_idx].size(0);
            
            // 从 host_storage_per_layer[layer_idx] 查出该用户的 KV 数据
            auto& host_kv = host_storage_per_layer[layer_idx].kv_data[uid];
            
            for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
                int gpu_page_id = page_ids[page_idx];
                int bytes = page_size * num_heads * head_dim * 2 * sizeof(float16);
                
                // 发起单个页的 H2D
                // 目标地址：gpu_cache_tables[layer_idx] + gpu_page_id * bytes
                void* gpu_dst = (void*)((uintptr_t)gpu_cache_tables[layer_idx].data_ptr()
                                        + gpu_page_id * bytes);
                void* host_src = host_kv.data() + page_idx * bytes;
                
                cudaMemcpyAsync(gpu_dst, host_src, bytes, cudaMemcpyHostToDevice, h2d_stream);
            }
        }
        
        // 该层 H2D 全部发起后，记录 event
        cudaEventRecord(layer_ready_events[layer_idx], h2d_stream);
    }
}
```

**性能考量**：
- 使用**独立 stream**（`h2d_stream`）进行异步 H2D，不阻塞计算 stream；
- 按层发起 H2D，便于**按层等待**（见 §4.2）；
- `cudaEventRecord` 记录该层完成点，供后续同步。

#### 4.1.2 等待 `onboard_wait` 与按层等待 `stream_wait_layer`

**非阻塞等待**（`kvcache_manager.py:110-125`）：

```python
def onboard_try_wait(self, kv_index_meta, task_handle):
    if self.host_kvstorage_manager.backend_name == "native":
        # native 真正支持非阻塞
        return self.host_kvstorage_manager.onboard_kvcache_wait(task_handle)
    elif self.host_kvstorage_manager.backend_name == "flexkv":
        # flexkv 暂未实现非阻塞，退化为阻塞 + 告警
        print("[WARNING] onboard_try_wait not implemented for flexkv")
        return self.onboard_wait(kv_index_meta, task_handle)
```

**按层等待的使用**（在 HSTU 注意力层，`paged_hstu_infer_layer.py:305-306`）：

```python
if kv_cache_metadata.kv_onload_handle is not None:
    kv_cache_metadata.kv_onload_handle.stream_wait_layer(self.layer_idx)
```

**HostKVTaskHandle 的实现**（`host_kvstorage_manager.py:74-76`）：

```python
def stream_wait_layer(self, layer_idx: int) -> None:
    if self.is_layerwise:  # native 后端置为 True
        self.handle.wait_layer(layer_idx)  # 调用 C++ 方法
```

**C++ wait_layer**（伪码）：

```cpp
// 在当前计算 stream 上等待该层的 H2D event
void wait_layer(int layer_idx) {
    // 获取当前 CUDA stream（推理 stream，与 h2d_stream 不同）
    cudaStream_t compute_stream = torch::cuda::current_stream().stream();
    
    // 在 compute_stream 上插入一个等待点：等到 layer_ready_events[layer_idx]
    cudaStreamWaitEvent(compute_stream, layer_ready_events[layer_idx], 0);
}
```

**重叠示意图**：

```
H2D Stream:     [Layer 0 H2D...]  [Layer 1 H2D...]  [Layer 2 H2D...]
                    ↓ event0          ↓ event1          ↓ event2
                    
Compute Stream: [L0 compute]  [等待 event0]  [L1 compute]  [等待 event1]  [L2 compute]
                                           ← 重叠 →                      ← 重叠 →
```

### 4.2 Offload (GPU→Host) 的生命周期

**关键区别**：offload 是**非阻塞异步回收**，与后续 post/MLP 重叠。

**代码流程**：`inference_dense_module.py:389-390` → `kvcache_manager.py:166-233` / `235-269`

#### 4.2.1 发起 `offload_launch`

```python
def offload_launch(self, index_meta, kvcache_metadata=None):
    # 1. native 后端：查出待下沉用户
    if self.host_kvstorage_manager.backend_name == "native":
        uids_to_offload = self.gpu_kvcache_mgr.check_for_offload(index_meta.user_ids)
        # check_for_offload 使用 LRU 返回「可驱逐」用户
    else:
        uids_to_offload = index_meta.user_ids
    
    # 2. 再查一次 host 已存长度（多 GPU 场景兼容）
    _index_meta = self.host_kvstorage_manager.build_index_meta(
        uids_to_offload,
        torch.empty(0, dtype=torch.int32))  # dummy seq lengths
    offloaded_lengths = self.host_kvstorage_manager.lookup_kvcache(_index_meta).host_cached_lengths
    
    # 3. 在 GPU 侧锁定待下沉的页
    (offload_user_ids, offload_start_indices, offload_page_indices_list) = \
        self.gpu_kvcache_mgr.acquire_offload_pages(uids_to_offload, offloaded_lengths, ...)
    
    if offload_user_ids.size(0) == 0:
        return None  # 无页可下沉
    
    # 4. 发起 D2H 传输
    task_handle = self.host_kvstorage_manager.offload_kvcache_launch(
        offload_user_ids, offload_start_indices, offload_page_indices_list,
        index_meta=index_meta, kvcache_metadata=kvcache_metadata)
    
    if task_handle and task_handle.handle is not None and \
       task_handle.status != HostKVTaskStatus.SKIPPED:
        self.ongoing_offload_tasks.append(task_handle)  # 记入待回收队列
        return task_handle
    else:
        # host 拒绝（如过载），立即释放 GPU 页锁
        self.gpu_kvcache_mgr.release_offload_pages(
            offload_user_ids, offload_start_indices,
            self.dummy_empty_tensor, offloaded=[0]*offload_user_ids.size(0))
        return None
```

**C++ offload_launch**（伪码）：

```cpp
void offload_kvcache_launch(torch::Tensor offload_uids, torch::Tensor offload_starts,
                            std::vector<torch::Tensor> page_lists,
                            KVOffloadHandle& handle) {
    
    int B = offload_uids.size(0);
    handle.task_ids.clear();
    
    // 为每一层分别发起 D2H
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        
        for (int seq_idx = 0; seq_idx < B; ++seq_idx) {
            int uid = offload_uids[seq_idx];
            int* gpu_page_ids = page_lists[seq_idx].data_ptr<int>();
            int num_pages = page_lists[seq_idx].size(0);
            int offload_start = offload_starts[seq_idx];  // 该用户在 host 已存长度
            
            // 关键：从 GPU 分页表读出新数据
            void* gpu_src = gpu_cache_tables[layer_idx].data_ptr<void>();
            int page_size_bytes = page_size * num_heads * head_dim * 2 * sizeof(float16);
            
            for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
                int gpu_page_id = gpu_page_ids[page_idx];
                void* gpu_page_addr = (void*)((uintptr_t)gpu_src + gpu_page_id * page_size_bytes);
                
                // 追加到 host 存储（扩展用户的 KV 向量）
                int host_offset = offload_start + page_idx * page_size;
                void* host_dst = host_storage_per_layer[layer_idx].kv_data[uid].data();
                host_dst = (void*)((uintptr_t)host_dst + host_offset * num_heads * head_dim * sizeof(float16));
                
                cudaMemcpyAsync(host_dst, gpu_page_addr, page_size_bytes, 
                               cudaMemcpyDeviceToHost, d2h_stream);
            }
        }
        
        // 同样记录 layer 事件
        cudaEventRecord(layer_ready_events[layer_idx], d2h_stream);
    }
    
    // 后续可异步等待这些 events
}
```

#### 4.2.2 轮询回收 `offload_try_wait`

**代码**：`kvcache_manager.py:235-269`

```python
def offload_try_wait(self):
    remain_tasks = []
    for task_handle in self.ongoing_offload_tasks:
        wait_result = self.host_kvstorage_manager.offload_kvcache_wait(task_handle)
        
        if wait_result.status == HostKVTaskStatus.LAUNCHED:
            # 仍在传输，保留在队列
            remain_tasks.append(task_handle)
        
        elif wait_result.status == HostKVTaskStatus.READY:
            # D2H 完成，整理后释放 GPU 页锁
            self.host_kvstorage_manager.finish_task(task_handle)
            uids, starts, lengths = self.host_kvstorage_manager.get_offload_handle_metadata(task_handle)
            self.gpu_kvcache_mgr.release_offload_pages(
                uids, starts, lengths, offloaded=[1]*uids.size(0))  # 标记成功
        
        elif wait_result.status == HostKVTaskStatus.SKIPPED:
            # 无需处理
            pass
        
        elif wait_result.status in (FAILED, TIMEOUT, CANCELLED):
            # 错误处理
            if self.host_kvstorage_fail_policy == "fail_close":
                raise RuntimeError(f"Offload failed: {wait_result.message}")
            else:  # fail_open
                # 保留 GPU 页数据，标记未成功 offload
                self.gpu_kvcache_mgr.release_offload_pages(
                    uids, starts, lengths, offloaded=[0]*uids.size(0))
    
    self.ongoing_offload_tasks = remain_tasks
```

**性能特性**：
- `offload_try_wait` 通常在 post/MLP 进行时调用（见 `inference_dense_module.py:389`），与计算重叠；
- 若 D2H 未完成，返回 `LAUNCHED` 继续轮询；
- 若完成，立即释放锁，下次推理可重用页。

---

## Part V：HSTU 注意力层中的 KV 操作

### 5.1 KV 追加写入 `append_kvcache`

**代码流程**：`paged_hstu_infer_layer.py:285-303`

```python
def forward_naive(self, batch_size, num_tokens, layer_input, jd, kv_cache_metadata):
    # ... 计算 Q, K, V ...
    key = key.view(-1, self._num_heads, self._attention_dim_per_head)  # [num_new_tokens, num_heads, head_dim]
    value = value.view(-1, self._num_heads, self._linear_dim_per_head)
    
    if kv_cache_metadata is not None:
        # 获取 GPU 分页表
        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)
        
        # 核心操作：把新 K,V 追加到分页表
        paged_kvcache_ops.append_kvcache(
            key,                                           # [num_new_tokens, num_heads, head_dim]
            value,
            kv_cache_metadata.batch_indices,              # [num_new_tokens]，每 token 属于哪个序列
            kv_cache_metadata.position,                   # [num_new_tokens]，每 token 在各序列中的位置
            jd.num_candidates_offsets[:batch_size+1],    # [batch_size+1]，候选集偏移
            kv_cache_metadata.new_history_nnz_cuda,      # GPU scalar，新 token 总数
            kv_cache_metadata.new_history_nnz,           # CPU int，新 token 总数
            paged_k_cache,                                # [num_pages, page_size, num_heads, head_dim]
            paged_v_cache,
            kv_cache_metadata.kv_indices,                # [num_pages]，页 ID（可能不连续）
            kv_cache_metadata.kv_indptr,                 # [batch_size+1]，页范围指针
            kv_cache_metadata.kv_last_page_len,          # [batch_size]，最后页长度
            0,  # NHD layout 标志
            self.num_sms)  # GPU SM 数量，用于 kernel 参数化
```

**append_kvcache C++ kernel**（伪码，核心逻辑）：

```cpp
// paged_kvcache_ops.cu
__global__ void append_kvcache_kernel(
    const float16* key,                // [num_new_tokens, num_heads, head_dim]
    const float16* value,
    const int* batch_indices,          // [num_new_tokens]
    const int* position,               // [num_new_tokens]，该 token 是序列中第几个
    const int* num_candidates_offsets, // [batch_size+1]
    float16** paged_k_cache,           // [num_pages][page_size][num_heads][head_dim]
    float16** paged_v_cache,
    const int* kv_indices,             // [num_pages]，页 ID
    const int* kv_indptr,              // [batch_size+1]，csr 指针
    int* kv_last_page_len,             // [batch_size]，输出/输入
    int num_heads, int head_dim, int page_size,
    int num_new_tokens) {
    
    // 启动 num_new_tokens 个 thread，每个处理一个新 token
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_new_tokens) return;
    
    int seq_idx = batch_indices[token_idx];      // 该 token 属于哪个序列
    int pos_in_seq = position[token_idx];        // 该 token 是序列中第几个
    
    // 1. 计算该 token 应该写到哪个页
    int page_idx_in_indptr = pos_in_seq / page_size;  // 在 kv_indptr 中的相对索引
    int page_id = kv_indices[kv_indptr[seq_idx] + page_idx_in_indptr];  // 物理页 ID
    int pos_in_page = pos_in_seq % page_size;    // 在页内的偏移
    
    // 2. 把该 token 的 K,V 写到分页表中
    for (int h = threadIdx.y; h < num_heads; h += blockDim.y) {
        for (int d = threadIdx.z; d < head_dim; d += blockDim.z) {
            // K
            paged_k_cache[page_id][pos_in_page][h][d] = 
                key[token_idx][h][d];
            // V
            paged_v_cache[page_id][pos_in_page][h][d] = 
                value[token_idx][h][d];
        }
    }
    
    // 3. 更新最后页长度
    if (page_idx_in_indptr == (kv_indptr[seq_idx+1] - kv_indptr[seq_idx] - 1)) {
        // 这是该序列的最后一页
        atomicMax(&kv_last_page_len[seq_idx], pos_in_page + 1);
    }
}
```

**关键设计点**：
1. **直接写分页表**：无需额外 buffer，直接在分页表中修改（kernel 内存合并友好）；
2. **batch 级并行**：num_new_tokens 个 threads 并行，充分 SM 利用；
3. **最后页长度维护**：`kv_last_page_len` 避免遍历未满的最后一页。

### 5.2 注意力读取与 page 参数传递

**代码**：`paged_hstu_infer_layer.py:307-327`

```python
jagged_attn_output = hstu_attn_varlen_func(
    query,                                       # [num_new_tokens, num_heads, head_dim]
    key, value,
    jd.seqlen_offsets[:batch_size+1],           # 输入序列偏移（新 token）
    kv_cache_metadata.kv_seqlen_offsets[:batch_size+1],  # 注意力读取偏移（历史+新）
    None, None,  # seqused_q, seqused_k
    jd.max_seqlen, jd.max_seqlen,
    jd.scaling_seqlen,
    None,  # num_contexts
    jd.num_candidates[:batch_size],             # 每序列候选集大小
    target_group_size=1,
    window_size=(-1, 0),
    alpha=self._alpha,
    # 关键：page 参数
    kv_cache=kv_cache_table,                    # 分页表
    page_offsets=kv_cache_metadata.kv_indptr,   # 页范围指针
    page_ids=kv_cache_metadata.kv_indices,      # 页 ID 列表
    last_page_lens=kv_cache_metadata.kv_last_page_len)
```

**HSTU attention kernel 如何读取 paged KV**（伪码）：

```cpp
// hstu_attn_kernel.cu
__global__ void hstu_attn_paged_kernel(
    const float16* Q,                    // [num_new_tokens, num_heads, head_dim]
    const float16* K_new,                // [num_new_tokens, num_heads, head_dim]
    const float16* V_new,
    const int* kv_seqlen_offsets,        // [batch_size+1]
    const float16** kv_cache_table,      // 分页表
    const int* page_offsets,             // [batch_size+1]，kv_indptr
    const int* page_ids,                 // [num_pages]，kv_indices
    const int* last_page_lens,           // [batch_size]，kv_last_page_len
    float16* output,                     // [num_new_tokens, num_heads, head_dim]
    ...) {
    
    int token_idx = blockIdx.x;  // 处理第 token_idx 个新 token
    int seq_idx = batch_indices[token_idx];  // 属于哪个序列
    int head_idx = blockIdx.y;
    
    // 1. 计算 query token 应该与多少个历史 token 交互
    int kv_len = kv_seqlen_offsets[seq_idx+1] - kv_seqlen_offsets[seq_idx];  // 历史 + 本 token
    
    // 2. 读取该序列的所有历史页
    float16 attn_output[HEAD_DIM] = {0};
    
    for (int kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        // 计算 kv_idx 对应的页位置
        int page_offset = page_offsets[seq_idx];         // 该序列第一页的索引
        int page_idx = kv_idx / PAGE_SIZE;
        int page_id = page_ids[page_offset + page_idx];  // 物理页 ID
        int pos_in_page = kv_idx % PAGE_SIZE;
        
        // 防止越界：最后一页可能不满
        if (page_idx == (page_offsets[seq_idx+1] - page_offsets[seq_idx] - 1)) {
            // 最后一页
            if (pos_in_page >= last_page_lens[seq_idx]) continue;
        }
        
        // 3. 从分页表读 K, V
        float16 k_val[HEAD_DIM];
        float16 v_val[HEAD_DIM];
        
        // 直接指针寻址（避免 gather）
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            k_val[d] = kv_cache_table[page_id][pos_in_page][head_idx][d];
            v_val[d] = kv_cache_table[page_id][pos_in_page][head_idx][d];  // V 同理
        }
        
        // 4. 计算注意力（QK^T 分数 + softmax + 加权 V）
        float qk_score = 0;
        for (int d = 0; d < HEAD_DIM; ++d) {
            qk_score += Q[token_idx][head_idx][d] * k_val[d];
        }
        qk_score *= ALPHA;  // 缩放
        
        // 后续做 softmax、加权聚合等...
    }
    
    // 5. 写输出
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        output[token_idx][head_idx][d] = attn_output[d];
    }
}
```

**性能特点**：
- **直接索引，无 gather**：利用 page_ids + pos_in_page 直接计算内存地址，无需额外 gather 操作；
- **内存合并**：相邻 threads 读取相邻页位置，合并友好；
- **避免 soft cache miss**：分页使得历史数据在 GPU 缓存中，访问延迟低。

---

## Part VI：CUDA Graph 与静态执行路径

### 6.1 为什么需要 CUDA Graph

**问题**：如上所述，metadata 依赖 batch 大小、序列长度等动态信息。但 CUDA Graph 需要**静态地址和参数**。

**方案**：预分配最大尺寸的 metadata buffer，运行时拷贝（`copy_kvcache_metadata`）。

**代码**：`inference_dense_module.py:233-242` / `kvcache_metadata.py:134-157`

```python
# 初始化阶段
max_batch_size = hstu_config.max_batch_size
max_num_tokens = hstu_config.max_seq_len * max_batch_size
self._kvcache_metadata = get_kvcache_metadata_buffer(
    batch_size=max_batch_size,
    num_new_tokens=max_num_tokens,
    num_pages=...)  # 静态最大 buffer
self._kvcache_metadata.kv_cache_table = [
    self.kvcache.gpu_kvcache_mgr.gpu_kvcache_tables[idx]
    for idx in range(hstu_config.num_layers)]

# 运行时，在 capture 或回放前拷贝
copy_kvcache_metadata(self._kvcache_metadata, kvcache_metadata)
```

### 6.2 分段 Capture 的设计（eager vs graph 路径）

**代码**：`hstu_block_inference.py:63-85` / `paged_hstu_infer_layer.py:361-474`

为了支持**按层等待** H2D（见 §4.1.2），CUDA Graph 不能一次性 capture 所有层。
相反，分层 capture：

```python
# 初始化阶段（setup_cudagraph）
for batch_size in [1, 2, 4, 8, ...]:
    for num_tokens in [32, 64, 128, ...]:
        # Segment 0: Layer 0 forward_input（投影 + append_kvcache）
        with torch.cuda.graph(graph[0]):
            uvqk = attention_layers[0].forward_input(..., static_metadata)
        
        # Segment 1: Layer 0 forward_output（attention）+ Layer 1 forward_input
        with torch.cuda.graph(graph[1]):
            output = attention_layers[0].forward_output(...)
            uvqk = attention_layers[1].forward_input(...)
        
        # ...继续
```

**回放时的同步**（`hstu_block_inference.py:157-161`）：

```python
for idx in range(0, num_layers - 1):
    self._hstu_graph[batch_size][num_tokens][idx].replay()
    
    if kv_cache_metadata is not None:
        # 在每次回放前，等待该层的 H2D
        kv_cache_metadata.kv_onload_handle.stream_wait_layer(idx - 1)
    
    self._hstu_graph[batch_size][num_tokens][idx + 1].replay()
```

**性能收益**：H2D 与层间计算完全重叠。

---

## Part VII：错误恢复与资源管理

### 7.1 Onboard 失败恢复

**代码**：`kvcache_manager.py:127-164`

```python
def onboard_wait(self, kv_index_meta, task_handle):
    wait_result = self.host_kvstorage_manager.onboard_kvcache_wait(task_handle)
    
    if wait_result.status in (FAILED, TIMEOUT, CANCELLED):
        # 1. 撤销 GPU 页分配（该用户分配的页标记为「未 onboard」）
        self.gpu_kvcache_mgr.revoke_onboard_pages(
            task_handle.user_ids,
            task_handle.metadata["onboard_start_indices"],
            task_handle.metadata["onboard_lengths"])
        
        # 2. 根据策略处理
        if self.host_kvstorage_fail_policy == "fail_close":
            raise RuntimeError(...)
        else:  # fail_open
            # 仅告警，推理继续；用户的旧 GPU 数据或 CPU 重算
            print(f"[WARNING] Onboard failed but continuing")
```

**revoke_onboard_pages C++ 逻辑**（伪码）：

```cpp
void revoke_onboard_pages(torch::Tensor uids, torch::Tensor starts, torch::Tensor lengths) {
    // 标记这些页为「未成功 onboard」
    // 下次 allocate 时可重用
    
    for (int i = 0; i < uids.size(0); ++i) {
        int uid = uids[i];
        int num_pages = (lengths[i] + page_size - 1) / page_size;
        
        auto it = user_to_pages.find(uid);
        if (it != user_to_pages.end()) {
            // 标记这些页的 onboard_state = FAILED
            for (int j = 0; j < num_pages; ++j) {
                int page_id = it->second.page_ids[j];
                page_onboard_state[page_id] = PageState::INVALID;
            }
        }
    }
}
```

### 7.2 Offload 失败与页锁管理

**代码**：`kvcache_manager.py:235-269`

```python
def offload_try_wait(self):
    for task_handle in self.ongoing_offload_tasks:
        wait_result = self.host_kvstorage_manager.offload_kvcache_wait(task_handle)
        
        if wait_result.status == HostKVTaskStatus.READY:
            # 成功下板，释放锁
            self.gpu_kvcache_mgr.release_offload_pages(
                ..., offloaded=[1, 1, ...])
        
        elif wait_result.status in (FAILED, TIMEOUT, CANCELLED):
            if self.host_kvstorage_fail_policy == "fail_open":
                # GPU 数据保留，标记未下板
                self.gpu_kvcache_mgr.release_offload_pages(
                    ..., offloaded=[0, 0, ...])
            else:
                raise RuntimeError(...)
```

**release_offload_pages C++ 逻辑**（伪码）：

```cpp
void release_offload_pages(torch::Tensor uids, torch::Tensor starts,
                           torch::Tensor lengths, std::vector<int>& offloaded) {
    for (int i = 0; i < uids.size(0); ++i) {
        int uid = uids[i];
        auto it = user_to_pages.find(uid);
        
        if (it != user_to_pages.end()) {
            for (int page_id : it->second.page_ids) {
                page_lock_flags[page_id] = false;  // 释放锁
                
                if (offloaded[i]) {
                    // D2H 成功，标记该用户 GPU 数据已同步到 host
                    page_synced_to_host[page_id] = true;
                } else {
                    // D2H 失败或跳过，GPU 数据仍可用于下次推理
                }
            }
        }
    }
}
```

---

## Part VIII：性能优化关键指标与 Tradeoff

### 8.1 内存占用与容量规划

```
GPU VRAM = num_layers * num_primary_pages * page_size * num_heads * head_dim * 2 * dtype_size

典型配置（A100）：
  num_layers = 12
  num_primary_pages = 4096
  page_size = 64
  num_heads = 32
  head_dim = 96
  dtype = bfloat16

GPU VRAM ≈ 12 * 4096 * 64 * 32 * 96 * 2 * 2 bytes ≈ 37.8 GB
```

### 8.2 延迟与吞吐 Tradeoff

| 操作 | 延迟 | 并发性 |
|------|------|--------|
| GPU lookup | ~1-5 us | 高（O(B) batch） |
| GPU allocate | ~100-500 us | 低（host 阻塞）|
| H2D onboard | 依赖带宽，~ms 级 | 高（async） |
| append_kvcache kernel | ~100 us | 高（num_tokens 并行）|
| HSTU attention | ~1-10 ms | 高（注意力并行）|
| D2H offload | 依赖带宽，~ms 级 | 高（async，后台） |

### 8.3 LRU 策略的命中率优化

**问题**：若 batch 中的用户频繁变化，LRU 可能驱逐刚用过的用户。

**改进方向**：
1. 分层 eviction policy（常热用户优先）；
2. 预热策略（推理前加载热用户）；
3. 动态页分配（热用户分配更多页）。

---

## Part IX：实战案例：从输入到输出的完整链路

```
输入：user_ids=[100, 101], seq_lengths=[512, 256], new_tokens=[8, 5]
batch_size=2, num_layers=12, page_size=64, device=0

【Phase 1】查询
  lookup_kvcache(user_ids, seq_lengths)
    → GPU lookup: user 100 cached 504 tokens, user 101 cached 251 tokens
    → Host lookup: user 100 cached 504 tokens, user 101 cached 251 tokens
    → merge: both cached fully or host longer
    → lookup_result.cached_lengths = [504, 251]

【Phase 2】分配
  allocate_kvcache
    new_hist_lengths = [512-504, 256-251] = [8, 5]
    num_new_tokens = 13
    num_total_pages = [ceil(512/64), ceil(256/64)] = [8, 4]
    C++ allocate:
      - LRU 检查：若 GPU 页不足，驱逐最老用户（假设用户 50）
      - 分配给用户 100: pages[p1, p2, ..., p8]（8 页）
      - 分配给用户 101: pages[q1, q2, ..., q4]（4 页）
      - 更新 kv_indptr = [0, 8, 12]，kv_last_page_len = [16, 4]（最后页有效长度）

【Phase 3】异步 onboard
  onboard_launch
    - 用户 100: host 有 504，GPU 也有 504，无需搬运
    - 用户 101: host 有 251，GPU 也有 251，无需搬运
    → 返回 SKIPPED（无数据需搬）

【Phase 4】剥离 token + embedding
  strip_cached_tokens：仅保留新 8 + 5 = 13 tokens
  sparse_module(embeddings)：仅 13 个 embedding

【Phase 5】HSTU 推理（以 Layer 0 为例）
  forward_naive(layer_input[13, 768], kv_cache_metadata)
    
    # Q, K, V 计算
    query[13, 32, 96]
    key[13, 32, 96]
    value[13, 32, 96]
    
    # append_kvcache：写新 KV 到分页表
    append_kvcache_kernel<<<...>>>(
        key[13, 32, 96], value[13, 32, 96],
        batch_indices=[0,0,0,0,0,0,0,0, 1,1,1,1,1],  // 13 个 token 的序列号
        position=[504,505,...,511, 251,252,253,254,255],  // 在序列中的位置
        paged_k_cache[pages][64][32][96],
        kv_indices=[p1, p2, ..., p8, q1, q2, q3, q4],
        kv_indptr=[0, 8, 12])
        
        对于 token 0（seq 0, pos 504）：
          page_idx = 504 / 64 = 7  (最后一页)
          page_id = kv_indices[0+7] = p8
          pos_in_page = 504 % 64 = 56
          写 key/value 到 paged_k_cache[p8][56][h][d]
    
    # attention 读取
    hstu_attn_varlen_func(
        query[13, 32, 96],
        kv_cache_table[pages][64][32][96],
        kv_seqlen_offsets=[0, 512, 512+256],  // 历史 + 新 token
        page_offsets=[0, 8, 12],
        page_ids=[p1, ..., p8, q1, q2, q3, q4],
        last_page_lens=[56, 4])
        
        对于 token 0（seq 0）：
          读取 512 个历史 token + 本 token = 513 个 KV
          从 8 个页（p1-p8）中读取
          注意：最后页 p8 仅有效位置 0-56（kv_last_page_len=56）
          
        计算 attention：Q[13] @ K[512+13] → softmax → 加权 V[512+13]

【Phase 6】Offload
  offload_try_wait：轮询前面的任务（若有），回收页
  offload_launch：
    check_for_offload：用户 50（被驱逐）页未 sync → 可下沉
    acquire_offload_pages：锁定用户 50 的页
    offload_kvcache_launch：从 GPU 搬 KV 到 host
    
【Phase 7】后处理 + 预测头
  postprocessor + MLP（此时 offload 可能还在后台运行）
  
【输出】logits[2, num_items]
```

---

## 总结与精通路径

### 核心洞察
1. **User-ID 哈希**：O(1) 查询，推荐系统专用
2. **分页结构**：GPU 内存按固定页管理，便于直接索引与 LRU 驱逐
3. **异步重叠**：H2D/D2H 与计算并行，降低端到端延迟
4. **按层同步**：stream_wait_layer 实现层间计算与上层 H2D 的完全重叠
5. **静态化**：CUDA Graph 依赖静态 buffer，runtime 拷贝元数据

### 精通阶段
- ✅ 理解内存布局（GPU 张量、host 存储、metadata buffer）
- ✅ 掌握异步链路（onboard/offload 的发起、等待、错误恢复）
- ✅ 深入 kernel 逻辑（append 如何写，attention 如何读 paged cache）
- ✅ 理解 tradeoff（容量 vs 延迟，LRU 驱逐时机）
- ✅ 能够调参或扩展（改进 fail policy、支持分布式等）
