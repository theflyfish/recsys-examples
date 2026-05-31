# KVCache 功能在代码流程中的应用（代码级详细分析）

> 本文档基于对仓库源码的逐行阅读编写，所有结论均给出 `文件:行号` 引用，便于核对。
> 适用范围：`corelib/recsys_kvcache_manager`（KV 缓存库）与 `examples/hstu`（HSTU 推理调用方）。

---

## 1. 设计目标与整体结构

`recsys_kvcache_manager` 为生成式推荐模型推理提供 **LLM 兼容的 KV 缓存**。其核心特点是
**按推荐系统用户 ID（user_id）缓存**，而不是像 LLM 那样按 token 前缀匹配
——因为推荐场景下不同用户的行为序列几乎没有公共前缀，按 user_id 查找是 O(1) 且更贴合业务
（见 `corelib/recsys_kvcache_manager/README.md:18-22`）。

### 1.1 类层次结构

```
KVCacheManager                       # 上层协调接口 (kvcache_manager.py)
 ├── GPUKVCacheManager               # GPU 分页 KV 表 (gpu_kvcache_manager.py)
 └── HostKVStorageManagerBase        # 主机/SSD/远端存储接口 (host_kvstorage_manager.py)
       ├── NativeHostKVCacheManager  # 纯 pinned host memory (native_host_kvcache_manager.py)
       └── FlexKVStorageManager      # 接入 FlexKV 多级存储 (flex_kvcache_manager.py)
```

- `KVCacheManager` 同时持有一个 `gpu_kvcache_mgr` 和一个 `host_kvstorage_manager`，
  在构造时把 GPU 缓存表注册给 host 端（`kvcache_manager.py:49-55`）。
- 底层 GPU 实现由 C++ 扩展 `kvcache_cpp.GPUKVCacheManagerImpl` 提供
  （`gpu_kvcache_manager.py:19,65-76`）；host 端由 `kvcache_cpp.HostKVStorageImpl` 提供
  （`native_host_kvcache_manager.py:20,61-71`）。Python 层主要负责编排与元数据组织。

### 1.2 GPU 缓存张量布局

GPU 缓存是一块连续张量，按层切分后再按 k/v 切分（`gpu_kvcache_manager.py:52-64`）：

```python
gpu_kvcache_tensor: [num_layers, num_primary_cache_pages, 2(k/v), page_size, num_heads, head_dim]
gpu_kvcache_tables = list(tensor.unbind(dim=0))   # 每层一个视图
# 层内再 unbind(dim=1) 得到 (paged_k_cache, paged_v_cache)
```

这是 **paged KV cache**：物理上以「页（page）」为单位管理，每页 `page_size` 个 token。
HSTU 注意力 kernel 可直接从该分页表读取，无需额外拷贝
（`README.md:42-43`）。

---

## 2. 关键数据结构

### 2.1 `KVLookupResult`（查找结果，`kvcache_utils.py:28-124`）

同时承载 GPU 与 host 两级缓存的命中信息，并提供 `merge()` 把两者合并：

| 字段 | 含义 |
|------|------|
| `gpu_cached_start_indices` / `gpu_cached_lengths` | GPU 中该用户已缓存的起始位置与长度 |
| `host_cached_start_indices` / `host_cached_lengths` | host 中该用户已缓存的起始位置与长度 |
| `cached_start_indices` / `cached_lengths` | merge 后的「综合已缓存」起始与长度 |

`merge()` 的合并规则（`kvcache_utils.py:46-124`）逐样本计算：
- host 无缓存 → 取 GPU 命中；
- GPU 无缓存 → 取 host 命中（并断言 host 从序列开头缓存，`host_cached_start_indices==0`）；
- 两者都有 → 取 `max(host_len, gpu_start+gpu_len)`，并要求 `gpu_start <= host_len`（不允许空洞）。

### 2.2 `KVCacheMetadata`（缓存元数据，`kvcache_metadata.py:22-59`）

描述「一个 batch 的分页 KV 状态」，是分配/写入/读取的核心载体。关键点：
**所有整型元数据共用一块扁平 buffer `metadata_gpu_buffer`，再切片成多个视图**
（`kvcache_metadata.py:83-110`），这样便于一次性管理、且对 CUDA Graph 友好：

```
metadata_gpu_buffer (int32, 长度 = 5*B + 4 + 2*num_new_tokens) 切片为：
  kv_indptr (B+1) | kv_last_page_len (B) | total_history_lengths (B)
  | total_history_offsets (B+1) | new_history_nnz_cuda (1)
  | new_history_offsets (B+1) | batch_indices (num_new_tokens) | position (num_new_tokens)
```

- `kv_indices`（即 `page_ids_gpu_buffer`）：分页表中的物理页 ID 列表。
- `kv_indptr`：每个序列在 `kv_indices` 中的页区间指针（CSR 风格）。
- `kv_last_page_len`：每序列最后一页的有效长度。
- `batch_indices` / `position`：新增 token 的 append 定位信息。
- `kv_seqlens` / `kv_seqlen_offsets`：**注意力实际读取长度** = 历史 + candidates（见 §4.3）。
- `kv_onload_handle`：异步 onboard（H2D）句柄，支持按层等待。

### 2.3 `KVIndexMeta`（`kvcache_utils.py:127-129`）

仅含 `user_ids` 与 `seq_lengths`，作为 host 端查找/索引的轻量句柄。

---

## 3. 配置入口

`KVCacheConfig`（`kvcache_config.py:22-62`）定义全部参数，经
`KVCacheManager.from_config()`（`kvcache_manager.py:352-380`）构造：
- 约束 `offload_chunksize % page_size == 0`（`kvcache_manager.py:354-356`）。
- 依据 `host_kvstorage_backend` 选择 `native` 或 `flexkv` 后端
  （`kvcache_manager.py:285-350`）。
- `offload_mode`（lazy/eager，`kvcache_utils.py:23-25`）、
  `host_kvstorage_fail_policy`（fail_open/fail_close）等策略在此注入。

在 HSTU 模型中，缓存对象在 `InferenceDenseModule.setup_for_kvcache()` 创建：
`self.kvcache = KVCacheManager.from_config(kvcache_config)`（`inference_dense_module.py:213-217`）。

---

## 4. 端到端推理流程（以 HSTU ranking 为例）

入口：`InferenceRankingGR.forward_with_kvcache()`（`inference_ranking_gr.py:137-177`）。
下面按实际调用顺序拆解，并标注每步的「重叠对象」。

### 4.1 顶层编排（`inference_ranking_gr.py:137-177`）

```python
# ① 查找：GPU + host 两级
index_meta, lookup_res = self.dense_module.kvcache.lookup_kvcache(
    user_ids, total_history_lengths)

# ② 在 GPU 分页表中分配页（含 LRU 驱逐）
kvcache_metadata = self.dense_module.kvcache.allocate_kvcache(index_meta, lookup_res)

# ③ 异步 onboard（host→GPU），与后续 strip + embedding 重叠
self.dense_module.kvcache.onboard_launch(index_meta, lookup_res, kvcache_metadata)

# ④ 剥离已缓存 token，仅保留新增 token + contextual
old_cached_lengths = lookup_res.cached_lengths
striped_batch = self.strip_cached_tokens(batch, old_cached_lengths)

# ⑤ 仅对新增 token 做 embedding 查找（计算量随之下降）
embeddings = self.sparse_module(striped_batch.features)

# ⑥ 进入 dense（HSTU + offload + MLP）
kvcache_info = (index_meta, lookup_res, kvcache_metadata)
logits = self.dense_module.forward_with_kvcache(
    striped_batch, embeddings, user_ids, total_history_lengths, kvcache_info)
```

### 4.2 ① 查找 `lookup_kvcache`（`kvcache_manager.py:66-78`）

```python
gpu_lookup_results = self.gpu_kvcache_mgr.lookup(user_ids)          # GPU 命中
index_meta = self.host_kvstorage_manager.build_index_meta(user_ids, sequence_lengths)
host_lookup_results = self.host_kvstorage_manager.lookup_kvcache(index_meta)  # host 命中
lookup_results = KVLookupResult.merge(gpu_lookup_results, host_lookup_results) # 合并
```
- GPU 端 `lookup` 调 C++ `impl_.lookup`（`gpu_kvcache_manager.py:89-95`）。
- native host 端 `lookup` 返回 host 缓存长度，起始恒为 0
  （`native_host_kvcache_manager.py:88-95`）。

### 4.3 ② 分配 `allocate_kvcache → GPUKVCacheManager.allocate`（`gpu_kvcache_manager.py:97-126`）

```python
new_hist_lengths = seq_hist_lengths - lookup_results.cached_lengths   # 真正需要新算的 token
num_new_tokens = sum(new_hist_lengths)
num_total_pages = sum(ceil(seq_hist_lengths / page_size))
output_kvcache_metadata = get_kvcache_metadata_buffer(batch_size, num_new_tokens, num_total_pages)
output_kvcache_metadata.kv_cache_table = self.gpu_kvcache_tables
self.impl_.allocate(uids, seq_hist_lengths, host_cached_lengths,
                    page_ids_gpu_buffer, metadata_gpu_buffer)   # C++：分页 + LRU 驱逐
```
- 若 GPU 页不足，由 C++ 实现按 **LRU** 驱逐最久未用用户的页（`README.md:42-43,71`）。
- 注意：当前实现 **驱逐时不做 offload**（`README.md:71`）。
- `allocate_kvcache` 是 **host 阻塞** 的，无法与其它操作重叠（`README.md:87`，已知限制）。

### 4.4 ③ 异步 onboard `onboard_launch`（`kvcache_manager.py:93-108`）

转发到 host 后端的 `onboard_kvcache_launch`。以 native 为例
（`native_host_kvcache_manager.py:97-150`）：
- 计算需要从 host 拉到 GPU 的区间：比较 `gpu_end = gpu_start+gpu_len` 与 `host_len`，
  只搬运 GPU 缺失而 host 有的部分（`native_host_kvcache_manager.py:103-117`）。
- 据 `kv_indptr/kv_indices` 取出目标 GPU 页列表，调用 `impl_.onload_kvcache(...)`
  发起异步 H2D，返回 `KVOnloadHandle`（`native_host_kvcache_manager.py:119-148`）。
- 若无数据需搬运，返回 `SKIPPED`（`native_host_kvcache_manager.py:129-137`）。
- 句柄写回 `kvcache_metadata.kv_onload_handle`（`kvcache_manager.py:104`），
  供后续 **按层等待**。native 后端 `is_layerwise=True`（`native_host_kvcache_manager.py:148`）。

### 4.5 ④ 剥离已缓存 token `strip_cached_tokens`（`inference_ranking_gr.py:86-135`）

- contextual 特征不进缓存，需扣除：`num_cached = clamp_min(origin_num_cached - num_context, 0)`
  （`inference_ranking_gr.py:89-91`）。
- 行为序列按 item/action 拆分缓存量
  （`inference_ranking_gr.py:92-94`）。
- 重建 `KeyedJaggedTensor`：仅保留未缓存的历史 token 与 contextual
  （`inference_ranking_gr.py:101-132`）。这样后续 embedding 与 HSTU 只处理「新增 token」。

### 4.6 ⑥ dense 前向 `forward_with_kvcache`（`inference_dense_module.py:331-395`）

```python
# 预处理：用 cached_lengths 作为新 token 的位置起点（位置编码对齐历史）
jagged_data = self._hstu_block._preprocessor(
    embeddings=embeddings, batch=batch,
    seq_start_position=kv_lookup_result.cached_lengths.cuda())   # :346-350

# 注意力实际读取长度 = 历史 + candidates
kvcache_metadata.kv_seqlen_offsets = total_history_offsets + num_candidates_offsets  # :354-357
kvcache_metadata.kv_seqlens       = total_history_lengths + num_candidates           # :358-360
kvcache_metadata.max_seqlen      += max_num_candidates                               # :361

# HSTU 计算（可选 CUDA Graph 路径，见 §5）
hstu_output = self._hstu_block.predict(batch_size, num_tokens, hidden, jd, kvcache_metadata)  # :371-387

# 先回收已完成的 offload，再发起本批 offload（与下面 post+MLP 重叠）
self.kvcache.offload_try_wait()                              # :389
self.kvcache.offload_launch(kv_index_meta, kvcache_metadata) # :390

# 后处理 + 预测头
jagged_data = self._hstu_block._postprocessor(jagged_data)   # :392
jagged_item_logit = self._mlp(jagged_data.values)            # :393
```

**关键纠正**：offload 在 dense 模块内、HSTU 计算之后、postprocessor+MLP 之前发起，
从而与 post/MLP 重叠；并非在顶层 ranking 模块。

### 4.7 注意力层内：KV 写入与读取（核心，`paged_hstu_infer_layer.py`）

KV 的 **写入** 和 **读取** 都发生在 **每个 HSTU 注意力层内部**，而不是由 manager 单独编排。
以 eager 路径 `forward_naive`（`paged_hstu_infer_layer.py:256-359`）为例：

```python
# 1) 线性投影得到 u,v,q,k；其中 k,v 是「新增 token」当前层的 KV
mixed_uvqk = self.uvqk_addmm_impl(normed_input, num_tokens)
(user, value, query, key) = torch.split(mixed_uvqk, self._split_arg_list, dim=-1)  # :274-283

if kv_cache_metadata is not None:
    kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
    (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)

    # 2) 把新增 token 的 k,v 追加进分页表（写）
    paged_kvcache_ops.append_kvcache(
        key, value,
        kv_cache_metadata.batch_indices, kv_cache_metadata.position,
        jd.num_candidates_offsets[:batch_size+1],
        kv_cache_metadata.new_history_nnz_cuda, kv_cache_metadata.new_history_nnz,
        paged_k_cache, paged_v_cache,
        kv_cache_metadata.kv_indices, kv_cache_metadata.kv_indptr,
        kv_cache_metadata.kv_last_page_len, 0, self.num_sms)            # :288-303

    # 3) 按层等待该层的 onboard(H2D) 完成，确保历史 KV 已就位
    if kv_cache_metadata.kv_onload_handle is not None:
        kv_cache_metadata.kv_onload_handle.stream_wait_layer(self.layer_idx)  # :305-306

    # 4) 注意力直接从分页表读取（历史KV + 刚写入的新KV）
    jagged_attn_output = hstu_attn_varlen_func(
        query, key, value,
        jd.seqlen_offsets[:batch_size+1],
        kv_cache_metadata.kv_seqlen_offsets[:batch_size+1],
        ... ,
        kv_cache=kv_cache_table,
        page_offsets=kv_cache_metadata.kv_indptr,
        page_ids=kv_cache_metadata.kv_indices,
        last_page_lens=kv_cache_metadata.kv_last_page_len)             # :307-327
```

要点：
- **写在读之前**：每层先 `append_kvcache` 写入新 token 的 KV，再 `stream_wait_layer`，
  再做 attention 读取（历史 + 新增）。
- `stream_wait_layer` 实现按层 H2D/计算重叠：`HostKVTaskHandle.stream_wait_layer`
  仅在 `is_layerwise` 时调用底层 `handle.wait_layer(layer_idx)`（`host_kvstorage_manager.py:74-76`）。
  native 后端支持，因此可把第 L 层 H2D 与前面层的计算重叠（`README.md:45-47`）。

---

## 5. 两条执行路径：eager 与 CUDA Graph

`HSTUBlockInference.predict()` 根据是否启用 graph 分流（`hstu_block_inference.py:63-85`）：

- **eager**：逐层调用 `forward_naive`（写 KV→按层等待→attention），见 §4.7。
- **CUDA Graph**：`forward_input`/`forward_output` 拆分以便分段 capture
  （`paged_hstu_infer_layer.py:361-474`）：
  - `forward_input`：做投影并 `append_kvcache` 写入（`:393-408`）；
  - `forward_output`：做 attention 读取（`:435-457`）；
  - 回放时在层间插入 `kv_onload_handle.stream_wait_layer(idx-1)`
    （`hstu_block_inference.py:158-161`）。
- CUDA Graph 下用 `copy_kvcache_metadata` 把动态 metadata 拷入静态 buffer
  （`inference_dense_module.py:369`、`kvcache_metadata.py:134-157`）。

---

## 6. Offload（GPU→host）异步链路

### 6.1 发起 `offload_launch`（`kvcache_manager.py:166-233`）

1. native 后端先 `check_for_offload` 选出可下沉用户，并再查一次 host 已存长度
   （`kvcache_manager.py:178-190`），以支持多 GPU 实例场景。
2. `acquire_offload_pages` 锁定待下沉的 GPU 页（按用户）
   （`kvcache_manager.py:195-204`、`gpu_kvcache_manager.py:203-209`）。
3. `host_kvstorage_manager.offload_kvcache_launch(...)` 发起异步 D2H
   （`kvcache_manager.py:209-216`）。
4. 若 host 端拒绝（如过载，返回 `None/SKIPPED`），立即释放页锁
   （`kvcache_manager.py:217-229`）。
5. 成功则记入 `ongoing_offload_tasks`（`kvcache_manager.py:232`）。

### 6.2 轮询回收 `offload_try_wait`（`kvcache_manager.py:235-269`）

非阻塞遍历进行中的任务：
- `LAUNCHED`：仍在传输，保留。
- `READY`：`finish_task` 完成，并 `release_offload_pages(..., offloaded=1)` 解锁 GPU 页。
- `SKIPPED`：跳过。
- `FAILED/TIMEOUT/CANCELLED`：按 `host_kvstorage_fail_policy`：
  - `fail_close` → 抛错；
  - `fail_open` → `cancel_task` 并以 `offloaded=0` 释放页（GPU 数据保留，不丢正确性）。

---

## 7. 错误处理与回退

### 7.1 onboard 失败（`kvcache_manager.py:127-164`）

`onboard_wait` 在 `FAILED/TIMEOUT/CANCELLED` 时：
- `revoke_onboard_pages` 撤销受影响页（`kvcache_manager.py:149-154`、`gpu_kvcache_manager.py:192-195`）；
- flexkv 后端按 `fail_close` 抛错 / `fail_open` 告警继续（`kvcache_manager.py:155-163`）。

`onboard_try_wait`（`kvcache_manager.py:110-125`）：native 真正非阻塞；
flexkv 当前未实现，会退化为阻塞 `onboard_wait` 并打印告警。

### 7.2 失败策略语义

`host_kvstorage_fail_policy`：
- `fail_open`（默认）：缓存子系统失败时尽量不影响推理正确性，回退到「重新计算/保留 GPU 数据」。
- `fail_close`：一旦失败立即抛错，便于在严格场景暴露问题。

---

## 8. 两种 host 后端对比（基于源码事实）

| 维度 | NativeHostKVCacheManager | FlexKVStorageManager |
|------|--------------------------|----------------------|
| 存储 | pinned host memory | 接入 FlexKV（CPU/本地/远端块） |
| 按层 onboard | 支持（`is_layerwise=True`） | 见各自实现 |
| `onboard_try_wait` | 真非阻塞 | 暂未实现，退化为阻塞（`kvcache_manager.py:117-121`） |
| 构造参数来源 | `kvcache_manager.py:289-306` | `kvcache_manager.py:307-346` |
| 已知限制 | 至多 1 个 GPU 管理器 + 1 个推理实例，需配合 user_id 路由隔离（`README.md:89-91`） | 客户端-服务器，可扩展多级 |

---

## 9. 已知限制（来自 `README.md:83-91`）

1. `allocate_kvcache` 为 host 阻塞，不能与其它操作重叠。
2. 每个 device 仅允许 **一个** GPU KV 管理器，且每个 GPU 管理器仅一个推理实例。
3. native host 后端同样限制为至多一个 GPU 管理器 + 一个推理实例，需配合 user_id 路由与实例隔离。

---

## 10. 一句话总结

KVCache 在本仓库的应用可概括为四段式：
**查找(GPU+host) → 分配(分页+LRU) → 异步 onboard 并剥离已缓存 token →
在每个 HSTU 层内 append 新 KV、按层等待 H2D、从分页表做 attention → 异步 offload 回收**。
其性能收益来自三处重叠：onboard 与 strip/embedding 重叠、按层 H2D 与逐层计算重叠、
offload 与 post/MLP 重叠；正确性回退由 fail_open/fail_close 策略与 revoke/release 路径保证。
```
