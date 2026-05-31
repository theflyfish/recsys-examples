# `hstu_attn_varlen_func` 深度剖析：参数、调用链路与新旧 QKV 融合机制

> 本文逐行剖析 HSTU 变长注意力函数 `hstu_attn_varlen_func`，覆盖：
> 1. 每个参数的精确语义（shape / dtype / 作用）
> 2. 从 Python 到 CUDA kernel 的完整调用链路
> 3. 一个可运行的 demo 用例
> 4. **核心**：新增 QKV 与历史 QKV 如何有机结合做 attention
>
> 所有结论均带 `文件:行号` 引用。

---

## Part 0：函数定位与背景

`hstu_attn_varlen_func` 是 HSTU（Hierarchical Sequential Transduction Unit）的**变长（variable-length）注意力**核心入口。它有两种工作模式：

1. **非 paged 模式**（训练/无缓存推理）：标准变长 attention，所有 K/V 来自输入张量。
2. **paged 模式**（KVCache 推理）：历史 K/V 来自分页缓存表 `kv_cache`，仅候选/目标 token 的 K/V 来自输入。

入口定义：`corelib/hstu/hstu_attn/hstu_attn_interface.py:185-279`

HSTU attention 与标准 Transformer attention 的关键区别（见参考实现 `test_paged_hstu_attn_kernel.py:130-165`）：

```python
qk_attn = einsum("bnhd,bmhd->bhnm", q, k)    # QK^T
masked_qk_attn = qk_attn * alpha             # ① 缩放（非 1/sqrt(d) 的 softmax）
masked_qk_attn = F.silu(masked_qk_attn)      # ② SiLU 激活替代 softmax！
masked_qk_attn = masked_qk_attn / scaling_seqlen  # ③ 按序列长度归一
masked_qk_attn = masked_qk_attn * mask       # ④ 因果/分组 mask
attn_output = einsum("bhnm,bmhd->bnhd", masked_qk_attn, v)  # 加权 V
```

**注意**：HSTU 用 `SiLU(α·QKᵀ)/scaling_seqlen` 替代了传统的 `softmax(QKᵀ/√d)`，这是 HSTU 论文的核心设计——更适合推荐序列建模。

---

## Part 1：参数精确语义

完整签名（`hstu_attn_interface.py:185-207`）：

```python
def hstu_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    num_contexts=None, num_targets=None, target_group_size=1,
    window_size=(-1, -1), alpha=1.0,
    rab=None, has_drab=False,
    kv_cache=None, page_offsets=None, page_ids=None, last_page_lens=None,
    cu_seqlens_t=None, func=None, scaling_seqlen=-1,
):
```

### 1.1 基础 QKV 张量

| 参数 | Shape | dtype | 语义 |
|------|-------|-------|------|
| `q` | `(total_q, nheads, headdim)` | bf16/fp16 | 所有序列的 query token **拼接**（变长，无 padding）。`total_q = Σ seqlen_q[i]` |
| `k` | `(total_k, nheads, headdim)` | bf16/fp16 | key token 拼接。**paged 模式下只含候选/目标部分有效** |
| `v` | `(total_k, nheads, headdim)` | bf16/fp16 | value token 拼接 |

约束（`:49-51`）：`q/k/v` 必须是 3 维 `(L, num_heads, head_dim)`。

### 1.2 变长索引（CSR 风格偏移）

| 参数 | Shape | 语义 |
|------|-------|------|
| `cu_seqlens_q` | `(batch_size+1,)` int32 | query 的**累积**序列长度。`cu_seqlens_q[i+1]-cu_seqlens_q[i]` = 第 i 个序列的 q 长度 |
| `cu_seqlens_k` | `(batch_size+1,)` int32 | key 的累积序列长度。**paged 模式下 = 历史长度 + 候选长度** |
| `max_seqlen_q` | int | batch 内最大 q 长度（kernel tiling 用） |
| `max_seqlen_k` | int | batch 内最大 k 长度 |

> 「cu」= cumulative。例如 3 个序列长度 `[5, 8, 3]` → `cu_seqlens = [0, 5, 13, 16]`。
> 这种 jagged/varlen 表示避免了 padding 浪费。

约束（`:250-253`）：`max_seqlen_q <= max_seqlen_k`（q 不能比 k 长，否则未定义）。

### 1.3 HSTU 特有的语义分段

HSTU 把每个序列切成三段：**context（上下文）+ history（历史行为）+ target（候选/目标）**。

| 参数 | Shape | 语义 |
|------|-------|------|
| `num_contexts` | `(batch_size,)` | 每序列的上下文 token 数。仅在非因果场景使用 |
| `num_targets` | `(batch_size,)` | 每序列的**目标/候选** token 数（要打分的 items） |
| `target_group_size` | int | 每个 target group 的 token 数（默认 1） |
| `cu_seqlens_t` | `(batch_size+1,)` | target 的累积偏移（可选） |

约束（`:238-249`）：
- `num_contexts != None` 时必须 `window_size==(-1,0)`（因果），否则未定义。
- `num_targets != None` 时同样必须因果。

### 1.4 注意力行为控制

| 参数 | 类型 | 语义 |
|------|------|------|
| `window_size` | `(left, right)` | 滑动窗口。`(-1,-1)`=全注意力；`(-1,0)`=**因果注意力** |
| `alpha` | float | QKᵀ 的缩放因子（替代 `1/√d`），典型 `1/√head_dim` |
| `scaling_seqlen` | int | attention 输出的归一化分母。`-1` 时取 `max_seqlen_q`（`:254-255`） |
| `rab` | `(batch, max_seqlen_k, max_seqlen_k)` | Relative Attention Bias，加到 QKᵀ 上 |
| `has_drab` | bool | 是否对 rab 求导（反向用） |

### 1.5 **Paged KVCache 四件套**（本文重点）

| 参数 | Shape | 语义 |
|------|-------|------|
| `kv_cache` | `(page_num, 2, page_size, nheads, headdim)` | 分页 KV 表。`dim=1` 的 `2` 分别是 K(0)/V(1) |
| `page_offsets` | `(batch_size+1,)` int32 | 每序列在 `page_ids` 中的**页范围**指针（CSR 风格） |
| `page_ids` | `(page_offsets[-1],)` int32 | 每序列用到的**物理页 ID** 列表（可不连续） |
| `last_page_lens` | `(batch_size,)` int32 | 每序列**最后一页**的有效 token 数（因为最后一页通常不满） |

> 这四个参数共同描述「历史 K/V 在分页表中的物理位置」。
> kernel 通过 `page_offsets[i] : page_offsets[i+1]` 找到序列 i 的页区间，
> 再用 `page_ids[...]` 解析出物理页，最后一页只取 `last_page_lens[i]` 个 token。

四者必须同时提供，kernel 才进入 paged 模式（`hstu_api.cpp:432`）：
```cpp
bool is_paged_kv = kv_cache.has_value() && page_offsets.has_value() 
                && page_ids.has_value() && last_page_lens.has_value();
```

### 1.6 返回值

```
out: (total_q, nheads, headdim)   # 与 q 同 shape 的注意力输出
```
内部（`:78,101`）：kernel 输出取前 `head_dim` 列，reshape 回 `(L, nheads, head_dim)`。

---

## Part 2：完整调用链路

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Python 用户层] paged_hstu_infer_layer.py:307                        │
│   hstu_attn_varlen_func(q, k, v, ..., kv_cache=, page_ids=, ...)     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ 参数校验 (:234-255)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Python 接口] hstu_attn_interface.py:257                             │
│   HstuAttnVarlenFunc.apply(...)   ← torch.autograd.Function          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ forward (:25-101)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [autograd.forward] hstu_attn_interface.py:55                         │
│   out, rab_padded = hstu_attn_cuda.varlen_fwd(...)                   │
│   import hstu_attn_2_cuda as hstu_attn_cuda  (:19)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ pybind11 调用
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [C++ 绑定] csrc/hstu_attn/hstu_api.cpp:335                           │
│   hstu_varlen_fwd(...)                                               │
│   ├─ is_paged_kv 判断 (:432)                                        │
│   ├─ set_params_fprop(...) 填充 Flash_fwd_params (:189-200)         │
│   │    params->kv_cache_ptr / page_ids / page_offsets / ...         │
│   └─ run_hstu_fwd<...>(params, stream) (:723 m.def 注册)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ INT_SWITCH(page_size) 模板分发 (hstu_fwd.h:781)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [CUDA Kernel] csrc/hstu_attn/src/hstu_fwd.h                          │
│   ├─ HstuBlockInfo 计算各段 offset (block_info.h:28-65)             │
│   ├─ 历史 K/V：从 kv_cache 按 page_ids 加载 (:446,578,585,630)      │
│   ├─ 候选 K/V：从输入 k/v 加载 (:582)                               │
│   ├─ QKᵀ → α 缩放 → SiLU → /scaling_seqlen → mask → ·V             │
│   └─ 写回 out                                                       │
└─────────────────────────────────────────────────────────────────────┘

[架构分发] Ampere(sm80) 走 csrc/hstu_attn；Hopper(sm90) 走 hopper/
  hstu import 在 examples/hstu/modules 中自动按 GPU arch 选择 (smoke test 注释)
```

### 2.1 Python 层参数重排（autograd → CUDA）

`HstuAttnVarlenFunc.forward`（`:25-77`）将参数透传给 `varlen_fwd`，顺序固定：

```python
out, rab_padded = hstu_attn_cuda.varlen_fwd(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, scaling_seqlen,
    num_contexts, num_targets, target_group_size,
    window_size[0], window_size[1],    # left, right 拆开
    alpha, rab, func,
    kv_cache, page_offsets, page_ids, last_page_lens,  # paged 四件套
    cu_seqlens_t,
)
```

### 2.2 C++ 层 paged 参数解析

`hstu_api.cpp:189-200`，从 `kv_cache` 张量的 stride 推导出布局：

```cpp
if (is_paged_kv) {
    params->kv_cache_ptr         = kv_cache.data_ptr();
    params->kv_cache_row_stride  = kv_cache.stride(-2);  // headdim 维 stride
    params->kv_cache_head_stride = kv_cache.stride(-3);  // page_size 维
    params->kv_cache_page_stride = kv_cache.stride(-4);  // 2(k/v) 维
    params->kv_cache_kvtensor_stride = kv_cache.stride(-5);  // page 维
    params->page_size  = kv_cache.size(-3);   // 每页 token 数
    params->total_pages = kv_cache.size(-5);  // 总页数
}
params->page_offsets   = static_cast<int*>(page_offsets);
params->page_ids       = static_cast<int*>(page_ids);
params->last_page_lens = static_cast<int*>(last_page_lens);
```

### 2.3 Kernel 层的关键 offset 计算

`block_info.h:28-65` 在每个 CUDA block 启动时计算该序列各段的物理偏移：

```cpp
struct HstuBlockInfo {
    sum_s_q       = cu_seqlens_q[bidb];          // 该序列 q 在拼接张量中的起点
    sum_s_page    = page_offsets[bidb];          // 该序列页区间起点
    actual_seqlen_q = cu_seqlens_q[bidb+1] - sum_s_q;   // q 总长（含候选）
    actual_seqlen_k = cu_seqlens_k[bidb+1] - sum_s_k;   // k 总长（历史+候选）
    actual_seqlen_t = num_targets[bidb];          // 候选/目标长度
    actual_page_num = page_offsets[bidb+1] - sum_s_page;  // 历史页数
    last_page_seqlen = last_page_lens[bidb];      // 最后页有效长度
};

// 候选 K/V 在输入张量中的偏移（关键！）
kv_cache_offset(row_stride) = 
    sum_s_q * row_stride + (actual_seqlen_q - actual_seqlen_t) * row_stride;
//  ↑ 序列起点          ↑ 跳过 new_history，定位到候选段起点
```

**这行注释道出真相**（`block_info.h:52`）：
> `// base: (new history + target), offset: new history`

即：输入 `q/k/v` 张量里，前 `new_history = actual_seqlen_q - actual_seqlen_t` 个是新历史 token，
后 `actual_seqlen_t` 个是候选 token。kernel 从 `new_history` 偏移处开始取候选 K/V。

---

## Part 3：新增 QKV 与历史 QKV 的有机结合（核心）

### 3.1 三段式序列结构

HSTU 推理时，每个用户序列在逻辑上是：

```
完整序列 = [已缓存历史] + [本次新增历史] + [候选/目标]
            └──────────┬──────────┘     └────┬────┘
              都进 KV Cache（持久）        不进 Cache（瞬时）
```

- **已缓存历史**：之前请求计算好的 K/V，存在 `kv_cache` 分页表里。
- **本次新增历史**：当前请求新到的历史行为，在 attention 之前已被 `append_kvcache` 写入分页表（`paged_hstu_infer_layer.py:288-303`）。
- **候选/目标**：要打分的 items，**不写入缓存**，K/V 留在输入张量里。

### 3.2 attention 时的 K/V 拼接逻辑（参考实现）

最清晰的说明在参考实现 `test_paged_hstu_attn_kernel.py:179-236`：

```python
def _hstu_paged_kv_attention(..., q, k, v, q_offsets, kv_cache, 
                             page_offsets, page_ids, last_page_lens, num_targets):
    k_con = empty(0, ...)   # 拼接后的完整 K
    v_con = empty(0, ...)

    for i in range(batch_size):
        page_num = page_offsets[i+1] - page_offsets[i]
        new_history_len = q_offsets[i+1] - q_offsets[i] - num_targets[i]  # 新历史长度

        # ① 历史 K/V：从分页缓存读取（满页）
        for j in range(page_num - 1):
            k_con = cat(k_con, kv_cache[page_ids[page_offsets[i]+j], 0])  # K
            v_con = cat(v_con, kv_cache[page_ids[page_offsets[i]+j], 1])  # V

        # ② 历史 K/V：最后一页（只取有效部分 last_page_lens[i]）
        k_con = cat(k_con, kv_cache[page_ids[page_offsets[i+1]-1], 0, :last_page_lens[i]])
        v_con = cat(v_con, kv_cache[page_ids[page_offsets[i+1]-1], 1, :last_page_lens[i]])

        # ③ 候选 K/V：从输入张量读取（跳过 new_history 段，取候选段）
        k_con = cat(k_con, k[q_offsets[i]+new_history_len : q_offsets[i+1]])
        v_con = cat(v_con, v[q_offsets[i]+new_history_len : q_offsets[i+1]])

    # 用拼好的完整 K/V 做 attention，Q 仍是全部输入 query
    return _hstu_attention_maybe_from_cache(q=q, k=k_con, v=v_con, ...)
```

**关键结论**：每个序列的注意力 K/V 序列 =
```
[缓存中的全部历史(满页)] ++ [缓存最后一页(部分)] ++ [输入中的候选K/V]
└────────────── 来自 kv_cache 分页表 ──────────────┘   └─ 来自输入 k/v ─┘
```

而 **Q 是全部输入 query**（新历史 + 候选都要算输出）。

### 3.3 为什么这样设计是「有机」的

1. **零冗余拷贝**：历史 K/V 一直躺在分页表，attention kernel 直接按 `page_ids` 寻址读取（`hstu_fwd.h:446`），不需要先 gather 到连续内存。

2. **新历史已经在缓存里**：注意 ③ 只取候选段 `k[new_history_len:]`，因为新历史段 `k[:new_history_len]` 在 attention 之前已被 `append_kvcache` 写进缓存了（成为 ① ② 的一部分）。所以输入张量里的新历史 K/V 是「冗余的」，kernel 直接忽略，转而用缓存版本。

3. **候选不污染缓存**：候选 items 是本次请求特有的，不应进缓存（下次请求的候选不同），所以留在输入张量，attention 用完即弃。

4. **因果 mask 保证正确性**：`window_size=(-1,0)` + `num_targets` 让 kernel 构造正确的 mask（`test:461-483`）——
   - 历史/新历史段：标准下三角因果（每个 token 看到自己及之前）。
   - 候选段：每个候选只看到「全部历史 + 自己」，候选之间互不可见（用 `eye` 单位矩阵，见 `test:476`）。这符合推荐打分语义：每个候选 item 独立打分，不能互相看到。

### 3.4 Kernel 内部的分块处理

kernel 把 K/V 序列按 `kBlockN` 分块（`hstu_fwd.h:109`）：

```cpp
const int n_block_paged = Paged_KV ? n_block_history : 0;  // 历史块数
```

遍历 K/V block 时：
- `n_block < n_block_paged`（历史区）：从分页表加载（`:578` `tVgV_page(..., page_ids[page_offset + n_block])`）。
- `n_block >= n_block_paged`（候选区）：从输入张量加载（`:582` `tVgV(..., n_block - n_block_paged)`）。

即 kernel 在**遍历过程中无缝切换数据源**，对上层完全透明。最后一页的边界用 `last_page_offset = kBlockN - last_page_seqlen`（`:105`）处理不满页。

---

## Part 4：可运行 Demo 用例

下面是一个最小可复现 demo（基于 `test_paged_hstu_attn_kernel.py` 简化），演示新旧 QKV 融合：

```python
import torch
from hstu import hstu_attn_varlen_func

def get_offsets(lengths):
    off = torch.zeros(len(lengths)+1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, 0, out=off[1:])
    return off

device = "cuda"
dtype = torch.bfloat16
num_heads, head_dim = 4, 128
page_size = 32

# ===== 场景：2 个用户 =====
#  User0: 历史 70 token  + 候选 10 token
#  User1: 历史 50 token  + 候选 8  token
batch_size = 2
history_lens = torch.tensor([70, 50], dtype=torch.int32, device=device)
num_candidates = torch.tensor([10, 8], dtype=torch.int32, device=device)

# ----- 1. 构造分页 KV 缓存（历史 K/V 已存在） -----
total_pages = 16
kv_cache = torch.zeros(total_pages, 2, page_size, num_heads, head_dim,
                       dtype=dtype, device=device)

page_ids, page_offsets, last_page_lens = [], [0], []
acc = 0
for i in range(batch_size):
    hlen = history_lens[i].item()
    npages = (hlen + page_size - 1) // page_size
    pids = list(range(acc, acc + npages))
    # 用随机数据填充历史 K/V（模拟之前请求 append_kvcache 写入的结果）
    for p, pid in enumerate(pids):
        tok = min(page_size, hlen - p*page_size)
        kv_cache[pid, :, :tok] = torch.randn(2, tok, num_heads, head_dim,
                                             dtype=dtype, device=device)
    page_ids += pids
    acc += npages
    page_offsets.append(acc)
    lpl = hlen % page_size
    last_page_lens.append(lpl if lpl > 0 else page_size)

page_ids       = torch.tensor(page_ids, dtype=torch.int32, device=device)
page_offsets   = torch.tensor(page_offsets, dtype=torch.int32, device=device)
last_page_lens = torch.tensor(last_page_lens, dtype=torch.int32, device=device)

# ----- 2. 构造输入 QKV -----
# q 长度 = 本次要算输出的 token = 候选（demo 简化：只对候选算输出）
# 但 k/v 输入需含 [新历史 + 候选]；这里假设无新历史，仅候选
seqlen_q = num_candidates.clone()              # 只对候选算 query
seqlen_k_input = num_candidates.clone()         # 输入 k/v 只有候选段
total_q = seqlen_q.sum().item()

q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)

cu_seqlens_q = get_offsets(seqlen_q)
# cu_seqlens_k = 历史长度 + 候选长度（attention 实际要看的 K 长度）
cu_seqlens_k = get_offsets(history_lens + num_candidates)

# ----- 3. 调用 paged attention -----
out = hstu_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=int(seqlen_q.max()),
    max_seqlen_k=4096,
    num_contexts=None,
    num_targets=num_candidates,        # 候选数
    target_group_size=1,
    window_size=(-1, 0),               # 因果
    alpha=1.0 / (head_dim ** 0.5),
    kv_cache=kv_cache,                 # 历史 K/V 来源
    page_offsets=page_offsets,
    page_ids=page_ids,
    last_page_lens=last_page_lens,
    scaling_seqlen=-1,
)
torch.cuda.synchronize()
print("output shape:", out.shape)     # (total_q, num_heads, head_dim)
# User0 的 10 个候选，每个都 attend 了缓存中的 70 个历史 token
# User1 的 8  个候选，每个都 attend 了缓存中的 50 个历史 token
```

**这个 demo 体现的融合**：
- 历史 K/V（70/50 token）从 `kv_cache` 分页表读取——**它们不在 `q/k/v` 输入里**。
- 候选 K/V（10/8 token）从输入 `k/v` 读取。
- kernel 自动把两者拼成完整 K/V 序列，让每个候选 query 与「全部历史 + 候选」做 attention。

> 完整带数值校验的版本见 `examples/hstu/test/test_paged_hstu_attn_kernel.py:367-538`，
> 它把 paged 模式输出与「手工拼接 K/V 的参考实现」对比，验证正确性。

---

## Part 5：实际推理中的完整时序（串联 KVCache）

回到 `paged_hstu_infer_layer.py:256-327` 的 `forward_naive`，看完整时序：

```python
# 第 layer_idx 层
mixed_uvqk = self.uvqk_addmm_impl(normed_input, num_tokens)   # 投影
(user, value, query, key) = split(mixed_uvqk, ...)            # 得到本层 Q,K,V

if kv_cache_metadata is not None:
    # ① 先把【新历史】的 K,V 写入分页缓存
    paged_kvcache_ops.append_kvcache(
        key, value,
        kv_cache_metadata.batch_indices, kv_cache_metadata.position,
        jd.num_candidates_offsets[:batch_size+1],   # 候选偏移：append 跳过候选
        kv_cache_metadata.new_history_nnz_cuda, kv_cache_metadata.new_history_nnz,
        paged_k_cache, paged_v_cache,
        kv_cache_metadata.kv_indices, kv_cache_metadata.kv_indptr,
        kv_cache_metadata.kv_last_page_len, 0, self.num_sms)

    # ② 按层等待该层历史 K/V 的 H2D onboard 完成
    if kv_cache_metadata.kv_onload_handle is not None:
        kv_cache_metadata.kv_onload_handle.stream_wait_layer(self.layer_idx)

    # ③ attention：历史(缓存) + 候选(输入) 融合
    jagged_attn_output = hstu_attn_varlen_func(
        query, key, value,
        jd.seqlen_offsets[:batch_size+1],              # cu_seqlens_q
        kv_cache_metadata.kv_seqlen_offsets[:batch_size+1],  # cu_seqlens_k = 历史+候选
        ...,
        num_candidates=jd.num_candidates[:batch_size], # num_targets
        kv_cache=kv_cache_table,                       # 分页表
        page_offsets=kv_cache_metadata.kv_indptr,
        page_ids=kv_cache_metadata.kv_indices,
        last_page_lens=kv_cache_metadata.kv_last_page_len)
```

**关键时序**：每层都是「**先写后读**」——
1. `append_kvcache` 把当前层新历史 K/V 写进缓存（与 onboard 的旧历史合并）。
2. `stream_wait_layer` 确保旧历史的 H2D 拷贝完成。
3. `hstu_attn_varlen_func` 做融合 attention。

`cu_seqlens_k` 的构造（`inference_dense_module.py:354-360`）正是融合的体现：
```python
kvcache_metadata.kv_seqlen_offsets = total_history_offsets + num_candidates_offsets
kvcache_metadata.kv_seqlens        = total_history_lengths + num_candidates
#                                    └─ 缓存历史 ─┘   └─ 输入候选 ─┘
```

---

## Part 6：常见陷阱与调试要点

| 陷阱 | 现象 | 排查 |
|------|------|------|
| `max_seqlen_q > max_seqlen_k` | ValueError(`:250`) | q 不能比 k 长；paged 下 k 含历史，一般不会触发 |
| paged 四件套缺一 | 退化为非 paged 模式 | 检查 `is_paged_kv`（`hstu_api.cpp:432`）四个都非 None |
| `last_page_lens` 算错 | attention 读到脏数据/越界 | 满页时应填 `page_size` 而非 0（见 `test:345`） |
| `num_targets` 与 `window_size` 冲突 | ValueError(`:242`) | 有 target 必须因果 `(-1,0)` |
| `append_kvcache` 漏写新历史 | 历史 attention 缺 token | 确认 append 在 attention 之前调用 |
| `kv_seqlen_offsets` 没加候选 | 候选 attend 不到自己 | 必须 `history_offsets + candidates_offsets` |
| build 不支持 paged | TORCH_CHECK 报错(`:187`) | 需编译支持 paged 的 hstu 版本（`test:30-33` 有断言） |

---

## 总结

### 一句话概括
`hstu_attn_varlen_func` 是 HSTU 的变长注意力入口，通过 `kv_cache/page_offsets/page_ids/last_page_lens` 四件套支持分页 KV 缓存：**历史 K/V 从分页表按页寻址读取，候选 K/V 从输入张量读取，kernel 在分块遍历中无缝拼接两者**，对每个 query token（新历史 + 候选）计算 `SiLU(α·QKᵀ)/scaling_seqlen` 加权的注意力。

### 融合机制三要点
1. **数据来源分离**：历史在缓存（持久），候选在输入（瞬时），kernel 按 block 切换数据源（`hstu_fwd.h:109,578,582`）。
2. **新历史先写后读**：`append_kvcache` 在 attention 前把新历史写进缓存，输入张量里的新历史段被 kernel 忽略（`block_info.h:52`）。
3. **mask 保语义**：因果 mask + 候选间单位矩阵 mask，保证每个候选独立 attend 全部历史（`test:461-483`）。

### 调用链
```
hstu_attn_varlen_func (Python)
  → HstuAttnVarlenFunc.apply (autograd)
    → hstu_attn_2_cuda.varlen_fwd (pybind)
      → hstu_varlen_fwd (hstu_api.cpp, set params)
        → run_hstu_fwd kernel (hstu_fwd.h, paged 寻址 + SiLU attention)
```
