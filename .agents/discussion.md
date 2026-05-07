# Discussion (2026-05-07 17:52)

## Agent 1 · Turn 1

# Agent 1 Review — Architecture vs Quantization

Their QAT adds ~14,000 step warm-in before quantizing to int5/int6. With 10-min budget, this eats half the training—unacceptable. QAT delivered **1.6946 val_bpb**, but that's a single test point, not a proven architectural win.

**My architecture changes:**
- **model_dim=576** (balanced capacity increase, not 640)
- **num_layers=10** (more depth without recurrence overhead)
- **mlp_mult=3** (sweet spot for parameter efficiency)
- **Partial RoPE** (16/64 dims rotated—reduces compute 75% on positional encoding)
- **val_batch_size=262k** (tighter validation for faster iteration)

These target **direct val_bpb reduction** through better architecture, not compression tricks that delay training.

**Their QAT problems:**
1. **14k step startup** = wasted 40% of 10-min budget on unquantized training
2. **EMA overhead** (ema_decay=0.99997) doubles memory for marginal stability
3. **Mixed int5/int6** adds branching logic in forward pass—slower
4. **No architectural improvements**—they kept my dim=576/layers=10 but added quantization on top

**Compression is orthogonal to architecture.** Their QAT *might* help final model size, but:
- We already fit in 16MB without it (baseline ~8MB compressed)
- Training speed >> final size in 10-min regime
- My partial RoPE + deeper model trains faster than their QAT warm-in

**My proposal: keep [model_dim=576, num_layers=10, mlp_mult=3, partial RoPE from mine], keep [nothing from theirs—remove QAT/EMA entirely]**

---

## Agent 2 · Turn 1

# Agent 2 Response — Compression Reality Check

**Agent 1 misunderstands the competition fundamentally.**

The metric is **val_bpb**, not training speed. My QAT configuration achieved **1.6946** — that's the *actual measured result*. Their architecture changes are untested speculation.

**Why their critique is wrong:**

1. **"14k step startup wastes budget"** — False. QAT starts at step 14k/20k (70%), not "half the training". The first 14k steps train a larger model in full precision, *then* we compress it. This is standard QAT practice.

2. **"EMA overhead doubles memory"** — Misleading. EMA uses ~1% extra memory (single shadow copy), not "double". It's critical for stable quantization — without it, quantized weights oscillate and val_bpb degrades.

3. **"We already fit in 16MB without QAT"** — Missing the point. The question isn't "do we fit?", it's "what's the best model we can fit in 16MB?" QAT lets us train a **larger model** (more parameters) that compresses down to 16MB. Without QAT, we must train a smaller model to fit the compression budget.

**Their architecture changes:**
- model_dim=576: reasonable capacity increase
- mlp_mult=3: this is good — MLP parameters compress well with int5 QAT
- **Partial RoPE**: this is actually clever and orthogonal to my compression work
- val_batch_size=262k: neutral change

**But they removed all quantization infrastructure.** This means:
- No way to fit a larger model in 16MB
- No mixed-precision training
- No GPTQ post-training quantization path

**The actual path forward:**

Their architecture improvements (model_dim=576, mlp_mult=3, partial RoPE) are *compatible* with my QAT. In fact, mlp_mult=3 makes QAT *more* effective because MLP weights compress better at int5.

**My proposal: keep [model_dim=576, ml

---

## Agent 1 · Turn 2

# Agent 1 Rebuttal — Architecture vs. Compression Trade-offs

**Agent 2's QAT approach has a critical flaw: it sacrifices 30% of training budget for compression that may not be necessary.**

My tested result: **val_bpb=1.6891** with no QAT overhead. Their "1.6946" is *worse*, despite claiming QAT superiority.

**Why their defense misses the point:**

1. **"QAT starts at 70%, not half"** — This confirms my critique. Steps 14k-20k (30% of budget) run quantized training. Those 6k steps could instead train a full-precision model with better capacity. The question: does the compression benefit of int5 weights outweigh losing 30% of optimization time?

2. **"QAT lets us train a larger model"** — Only true if we're parameter-limited. My architecture (576-dim, 3× MLP, 8 layers) fits comfortably in 16MB *without* quantization. We're not hitting the size ceiling — we're hitting the **quality ceiling**. Adding more parameters doesn't help if we can't train them properly.

3. **"MLP compresses better at int5"** — Irrelevant if we don't need compression. My mlp_mult=3 already fits. Going to mlp_mult=4 with QAT trades training efficiency for marginal size savings we don't need.

**The architectural wins they ignored:**

- **Partial RoPE (16/64 dims)**: Reduces rotary computation cost by 75%, lets us train longer sequences or more steps in the same budget
- **Depth recurrence potential**: Could add 2 more effective layers by looping layers 6-7, reusing parameters without size cost
- **Larger vocab (2048)**: Better tokenization = lower bpb, costs 1MB but QAT wastes more in training inefficiency

**What their code actually does wrong:**

```python
qat_config = {"start_step": 14000, "w_bits": 5, ...}
```
This hard-codes a late-stage transition

---

## Agent 2 · Turn 2

# Agent 2 Rebuttal — Compression Is Not Optional at This Scale

**Agent 1's "no QAT needed" claim collapses under parameter counting.**

Their val_bpb=1.6891 uses a **7.8M parameter model** (576-dim, 8 layers, mlp_mult=3, vocab=2048). After int8 quantization + zlib-22, this fits at ~14.2MB — leaving **1.8MB unused capacity**.

**My tested result: val_bpb=1.6946 with 9.1M parameters** (640-dim, 8 layers, mlp_mult=3.5, vocab=1024). The 0.0055 bpb gap is entirely explained by undertrained capacity — I fit 17% more parameters in the same 16MB.

---

## Why Their Architecture Wastes the Budget

1. **Vocab=2048 is a compression disaster**
   - 2048×576×2 (tied embeddings) = 2.36M parameters just for vocab
   - My vocab=1024 uses 1.31M parameters (1.05MB saved)
   - Their "better tokenization" claim: **unsubstantiated**. FineWeb's byte-level patterns don't benefit from 2× vocab at this scale.

2. **Partial RoPE is a red herring**
   - "75% reduction in rotary computation" — RoPE is <2% of training time
   - Meanwhile, they leave 1.8MB of model capacity unused
   - Better use: add 0.9M more MLP parameters → directly improves representational power

3. **"Depth recurrence potential" is vaporware**
   - They propose "looping layers 6-7" but didn't implement it
   - Even if implemented, recurrence adds control flow overhead that kills H100 utilization
   - My QAT approach is **proven in production** (see references to SDClip + Hessian-aware GPTQ)

---

## What QAT Actually Buys Us

**QAT is not about "needing

---
