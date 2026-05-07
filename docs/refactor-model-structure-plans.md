# Model Architecture Refactoring Plan (Revised)

## Executive Summary

**My initial assessment was partially wrong.** The examples you provided (crafting tables with completely different tools, ores with different crystal shapes, sprite outline changes) prove that this is **not** a deterministic pixel-to-pixel mapping. Resource packs redesign content, not just recolor it. This IS a generative task requiring the model to infer style patterns and apply them creatively to unseen blocks.

**However**, the current architecture still fails for fixable reasons: pack embeddings are catastrophically under-conditioned, the model is 6x larger than necessary, and 50 timesteps is overkill for 32x32 images.

**Revised primary recommendation**: Small per-pack diffusion models (~5M parameters, 10-20 timesteps) with **explicit style reference images** (not pack embeddings). The model sees 1-3 example textures from the target pack during both training and inference, using cross-attention to copy style patterns. Train one model per pack.

---

## 1. Why the Current Model Fails (Revised Analysis)

### 1.1 Pack Embeddings Are Catastrophically Under-Conditioned

Your images prove that a 256-dimensional vector cannot encode style. Consider:
- Pack A draws **scissors and saws** on crafting tables
- Pack B draws **hammers and wrenches** 
- Pack C draws **completely different tools**

A vector has no way to represent "what tools does this pack prefer?" or "how does this pack redraw ore crystals?" The model needs to **see actual examples** from the target pack to know what style to apply.

**The original roadmap was correct**: style reference images are essential. Removing them in favor of pack embeddings was the critical mistake.

### 1.2 Model Is 6x Too Large

Current model: ~30M parameters.
Needed: ~5M parameters.

With only 50-200 training pairs per pack, a 30M-parameter model memorizes training noise rather than learning generalizable style patterns. Overfitting is guaranteed.

### 1.3 50 Timesteps Is Excessive

For 32x32 pixel art:
- 50 timesteps = 50 forward passes per image
- The "noise" is just masked tokens, not Gaussian noise
- 10-20 timesteps with a good scheduler (DDIM, DPM++) is sufficient
- Reduces training time by 60-80%

### 1.4 Discrete Tokenization Is Acceptable (With Caveats)

My initial criticism of tokenization was overstated. For pixel art:
- A 256-color palette CAN capture most pack styles if built per-pack or group
- The issue isn't tokenization itself, but the **shared palette across all packs**
- Each pack should have its own palette, OR we use continuous RGB outputs

### 1.5 Multi-Pack Training Creates Interference

Training one model on 15+ packs simultaneously forces it to learn:
- "Crafting tables might have scissors OR saws OR hammers..."
- The model can't commit to any pack's specific style
- Per-pack training eliminates this ambiguity

---

## 2. The Real Nature of the Problem

### 2.1 This Is Few-Shot Style Transfer

Given:
- Content: Vanilla texture of block X (never seen in target pack)
- Style refs: 1-3 textures from target pack (blocks A, B, C)
- Goal: Generate block X in target pack's style

The model must learn analogies:
- "If vanilla stone -> Mythic stone looks like this..."
- "And vanilla crafting table -> Mythic crafting table looks like this..."
- "Then vanilla diamond_ore -> Mythic diamond_ore should look like..."

This requires:
1. Content understanding (what is this block structurally?)
2. Style understanding (what does this pack do to textures?)
3. Pattern transfer (apply style patterns to new content)

### 2.2 Why Per-Pack Models Are Better

| Aspect | Shared Multi-Pack | Per-Pack |
|--------|-------------------|----------|
| **Style ambiguity** | Must disambiguate 15+ styles simultaneously | Learns one style only |
| **Pattern specificity** | "Tools on crafting table could be anything" | "This pack uses scissors" |
| **Training stability** | Pack embeddings fight for control | Single conditioning signal |
| **Data per pack** | 50-200 examples shared across packs | 50-200 examples focused on one style |
| **Convergence** | Slow, unstable | Fast, stable |
| **Debuggability** | Hard to tell which pack is failing | Easy: test each model independently |

### 2.3 What About Generalization?

You correctly noted the model must generalize to unseen blocks. Per-pack models handle this because:
- They learn the pack's style transformation from training pairs
- During inference, they apply the learned transformation to new vanilla blocks
- They don't memorize specific blocks; they learn "how this pack stylizes textures"

Example: If the model sees 50 (vanilla, Mythic) pairs during training, it learns Mythic's style. At test time, given vanilla `redstone_ore` (unseen), it applies the learned Mythic style.

---

## 3. Revised Architecture: Small Per-Pack Diffusion with Style References

### 3.1 Core Design

```python
class StyleAwareUNet(nn.Module):
    """
    Small U-Net (~5M parameters) for 32x32 pixel-art style transfer.
    Uses explicit style reference images via cross-attention.
    Designed for RTX 4060 Laptop (8GB VRAM).
    """
    def __init__(self, vocab_size=257, style_channels=64, base_channels=128):
        super().__init__()
        
        # Content encoder: processes vanilla target block
        self.content_embed = nn.Embedding(vocab_size, 64)
        self.content_encoder = nn.Sequential(
            nn.Conv2d(64, base_channels, 3, padding=1),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),  # 16x16
            ResBlock(base_channels*2),
            nn.Conv2d(base_channels*2, base_channels*2, 4, 2, 1),  # 8x8
            ResBlock(base_channels*2),
        )
        
        # Style encoder: processes 1-3 reference textures from target pack
        self.style_embed = nn.Embedding(vocab_size, 64)
        self.style_encoder = nn.Sequential(
            nn.Conv2d(64, style_channels, 3, padding=1),
            ResBlock(style_channels),
            nn.Conv2d(style_channels, style_channels, 4, 2, 1),  # 16x16
            ResBlock(style_channels),
            nn.Conv2d(style_channels, style_channels, 4, 2, 1),  # 8x8
            ResBlock(style_channels),
        )
        
        # Cross-attention: content queries attend to style keys/values
        self.cross_attn = CrossAttention(
            dim=base_channels*2,
            context_dim=style_channels,
            num_heads=4
        )
        
        # Time embedding (for diffusion)
        self.time_embed = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, base_channels*2),
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            ResBlock(base_channels*2),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            # 16x16 -> 32x32
            ResBlock(base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, vocab_size-1, 1),  # Predict tokens
        )
    
    def forward(self, noisy_target, content_source, style_refs, t):
        # noisy_target: [B, H, W] masked tokens
        # content_source: [B, H, W] vanilla tokens (for structure)
        # style_refs: [B, N, H, W] N reference textures from target pack
        # t: [B] diffusion timesteps
        
        # Encode content
        content = self.content_embed(content_source).permute(0, 3, 1, 2)
        content_feat = self.content_encoder(content)  # [B, C, 8, 8]
        
        # Encode style references (average pool across N refs)
        B, N, H, W = style_refs.shape
        style_refs = style_refs.view(B*N, H, W)
        style = self.style_embed(style_refs).permute(0, 3, 1, 2)
        style_feat = self.style_encoder(style)  # [B*N, C, 8, 8]
        style_feat = style_feat.view(B, N, *style_feat.shape[1:])
        style_feat = style_feat.mean(dim=1)  # [B, C, 8, 8]
        
        # Cross-attention: content attends to style
        content_feat = self.cross_attn(content_feat, style_feat)
        
        # Add time embedding
        time_emb = self.time_embed(timestep_embedding(t, 64))
        content_feat = content_feat + time_emb[:, :, None, None]
        
        # Decode
        output = self.decoder(content_feat)
        return output
```

**Key features**:
- **~5M parameters** (down from 30M)
- **Cross-attention** between content and style references (not pack embeddings)
- **Content source** provides structural guidance ("this is a crafting table")
- **Style references** provide actual examples ("this pack draws scissors")
- **No self-attention** (unnecessary for 32x32, saves VRAM)

### 3.2 Style Reference Selection

During training and inference:
1. For target block X, sample 1-3 other blocks from the same pack
2. These are the style references
3. The model sees: "Given vanilla X and examples A,B,C from pack Y, generate Y's version of X"

**Why this works**: The model can look at the references and copy patterns:
- "Reference A shows this pack uses desaturated colors"
- "Reference B shows this pack adds dark outlines"
- "Reference C shows this pack's ore crystals are diamond-shaped"
- "I'll apply these patterns to the vanilla content"

### 3.3 Timestep Reduction

Use **DDIM sampler** with 10-20 timesteps instead of 50:
```python
# Training: random t in [1, 20]
t = torch.randint(1, 20, (B,))
noisy = apply_mask(target, t, T=20)

# Inference: DDIM with 20 steps
for t in range(20, 0, -1):
    logits = model(noisy, content, style_refs, t)
    # Predict and unmask most confident tokens
```

**Training time reduction**: ~60% faster.
**Inference time reduction**: 2.5x faster (50 steps -> 20 steps).

---

## 4. Data Pipeline Changes

### 4.1 Per-Pack Datasets

Instead of one multi-pack dataset, create one dataset per pack:

```python
class PackStyleDataset(Dataset):
    def __init__(self, pack_name, split="train"):
        self.pairs = load_pairs(pack_name, split)  # [(vanilla, target), ...]
        self.pack_textures = load_all_pack_textures(pack_name)  # For style refs
    
    def __getitem__(self, idx):
        vanilla, target = self.pairs[idx]
        
        # Sample 1-3 random textures from same pack as style references
        style_refs = random.sample(self.pack_textures, k=random.randint(1, 3))
        
        return {
            "content": vanilla,        # Vanilla version of target block
            "target": target,          # Target pack version
            "style_refs": style_refs,  # Other blocks from target pack
        }
```

### 4.2 Continuous RGB vs Discrete Tokens

**Recommendation**: Use continuous RGB (not tokens) for the style references, but keep discrete tokens for the target prediction.

Why:
- Style references need to show actual colors/patterns (continuous)
- Target prediction can be discrete (palette indices) for pixel-art purity
- OR: Predict continuous RGB and snap to palette post-processing

Simpler approach: **Predict continuous RGB for everything**.
- Input: vanilla RGB 32x32
- Style refs: pack RGB 32x32
- Output: pack RGB 32x32
- Loss: L1 + MS-SSIM
- Post-process: snap to nearest palette color (optional)

This avoids the complexity of discrete diffusion entirely while preserving pixel-art aesthetics.

### 4.3 Handling Sprites and Alpha

For items/sprites with transparency:

```python
# Predict RGBA (4 channels)
class RGBAStyleUNet(StyleAwareUNet):
    def __init__(self, ...):
        super().__init__(...)
        self.decoder[-1] = nn.Conv2d(base_channels, 4, 1)  # RGBA output
    
    def forward(self, ...):
        rgb_logits = super().forward(...)
        # Split into RGB and Alpha
        rgb = rgb_logits[:, :3, ...]
        alpha = torch.sigmoid(rgb_logits[:, 3:4, ...])  # [0, 1]
        return rgb, alpha
```

**Training**:
- Loss = L1(RGB_pred, RGB_target) + BCE(Alpha_pred, Alpha_target)
- Alpha channel tells the model where the sprite outline is
- The model learns to preserve or transform outlines based on style refs

---

## 5. Training Strategy

### 5.1 Per-Pack Training

```bash
# Train one model per pack
for pack in Mythic Faithful Patrix Chibi; do
    python train.py \
        --model style_aware_unet \
        --pack $pack \
        --steps 10000 \
        --batch_size 4 \
        --lr 3e-4 \
        --timesteps 20 \
        --save_dir checkpoints/$pack
done
```

**Expected training time**: 10-20 minutes per pack on RTX 4060 Laptop.

### 5.2 Why Per-Pack Converges Faster

1. **Single style objective**: The model doesn't need to learn "which pack am I generating?"
2. **Relevant style refs**: All style references are from the same pack
3. **No pack embedding confusion**: Cross-attention focuses on actual textures
4. **Appropriate capacity**: 5M params is enough to learn one pack's style, not too much to overfit

### 5.3 Sanity Checks

Before full training:
1. **Overfit one pair**: Train on a single (vanilla, target) pair with style refs. Should converge in < 500 steps.
2. **Style ref ablation**: Train with and without style references. Without refs should fail (proves refs are necessary).
3. **Generalization test**: Hold out 5 blocks from training. Evaluate on held-out blocks after training.

---

## 6. Evaluation of Approaches

### 6.1 Approach 1: Small Per-Pack Diffusion with Style References (RECOMMENDED)

**Pros**:
- Explicit style references solve the under-conditioning problem
- Cross-attention can copy complex patterns (tools, crystals, outlines)
- Per-pack training is stable and fast
- Generalizes to unseen blocks within a pack
- 10-20 timesteps is feasible on 4060

**Cons**:
- Need to store N models for N packs (~25MB each = manageable)
- Inference requires sampling style refs from target pack
- Still requires iterative sampling (though fast)

**Best for**: Your use case. Handles pattern changes, generalizes to new blocks, explicit style control.

### 6.2 Approach 2: Base Model + Per-Pack LoRA

**Architecture**:
- Train one base StyleAwareUNet on ALL packs
- Add LoRA adapters to cross-attention and FiLM layers
- Each pack gets a ~200KB LoRA adapter

**Training**:
1. Phase 1: Train base model on all packs (joint training, harder)
2. Phase 2: Freeze base, train LoRA per pack (fast)

**Pros**:
- Small per-pack footprint (~200KB vs ~25MB)
- Unified inference code
- Can add new packs quickly

**Cons**:
- Base model training is unstable (same pack interference issues)
- LoRA may not capture complex style changes (like completely different tools)
- If base model is bad, all LoRAs are bad

**Verdict**: Good for Phase 2, but train per-pack models first to prove the concept.

### 6.3 Approach 3: Stable Diffusion + LoRA (Last Resort)

**Pros**:
- Extremely powerful, handles complex generation
- Large pre-trained base understands textures, lighting, materials
- LoRA training is well-documented

**Cons**:
- **Very slow on 4060**: 2-5 seconds per texture (vs 50ms for small diffusion)
- **Resolution mismatch**: SD trained on 512x512, may distort 32x32 pixel art
- **Hallucination risk**: May add non-existent details or change block identity
- **Data requirements**: Needs 100+ examples per pack for good LoRA
- **Overkill**: Using a 4B parameter model for 32x32 textures is absurd

**Verdict**: Only if Approaches 1-2 fail completely. Not recommended for production use on 4060.

---

## 7. Why the Original Plan Was Wrong

### 7.1 Deterministic CNN Cannot Handle Pattern Changes

A direct CNN mapping `vanilla -> target` assumes:
- Pixel (10, 15) in vanilla maps to pixel (10, 15) in target
- Only colors change, not content

Your images show this is false:
- Crafting table contents are completely rearranged
- Ore crystals have different shapes and positions
- A CNN would try to preserve vanilla structure, failing to capture the pack's redesign

### 7.2 Diffusion Is Actually the Right Paradigm (Just Poorly Implemented)

Diffusion IS appropriate because:
- It allows the model to iteratively refine the output
- It can change content/patterns, not just colors
- It handles the stochastic nature of "what tools go on this crafting table?"

The current implementation fails because:
- Pack embeddings are too weak (should be style references)
- Model is too large (30M -> 5M)
- Too many timesteps (50 -> 20)
- Multi-pack training causes interference

### 7.3 Per-Pack Is Correct, But Needs Generation

Per-pack models are still the right choice, but they need to be generative (diffusion-based), not deterministic (CNN-based).

---

## 8. Implementation Roadmap

### Phase 1: Prove Style References Work (1-2 days)

1. **Implement StyleAwareUNet** (~3 hours)
   - Content encoder, style encoder, cross-attention
   - ~5M parameters
   - Test forward pass

2. **Implement per-pack dataset with style refs** (~2 hours)
   - Load pack textures
   - Sample style references
   - Return (content, target, style_refs) tuples

3. **Train on one pack (e.g., Mythic)** (~2 hours)
   - 10k steps, 20 timesteps
   - Evaluate on held-out blocks
   - Verify generalization

4. **Ablation: train without style refs** (~1 hour)
   - Should perform much worse
   - Proves cross-attention + refs are essential

**Success criteria**:
- Training loss converges to < 1.0
- Validation pixel accuracy > 40%
- Visual inspection: generated textures match pack style
- Held-out blocks show correct style patterns

### Phase 2: Scale to All Packs (2-3 days)

1. **Batch training script** (~3 hours)
   - Train all packs sequentially
   - Save per-pack checkpoints

2. **Evaluation suite** (~3 hours)
   - Generate validation matrices
   - Compute per-pack metrics
   - Side-by-side comparisons

3. **Sprite/alpha support** (~2 hours)
   - Add RGBA output
   - Handle transparency in preprocessing
   - Evaluate on items (apple, bow, etc.)

### Phase 3: Optimization (Optional, 3-5 days)

1. **Base model + LoRA** (~2 days)
   - Distill per-pack models into base + adapters
   - Verify LoRA captures style patterns

2. **ONNX export** (~1 day)
   - Export to ONNX for faster inference
   - Optional: INT8 quantization

3. **DDIM / DPM++ sampler** (~1 day)
   - Implement faster samplers
   - Reduce inference to 10 steps

---

## 9. Hardware Considerations (RTX 4060 Laptop)

### 9.1 VRAM Budget

| Component | Current Model | StyleAwareUNet |
|-----------|--------------|----------------|
| Model parameters | ~30M (120MB) | ~5M (20MB) |
| Activations (bs=4) | ~4GB | ~500MB |
| Style refs (3 refs) | N/A | ~200MB |
| Gradients + Optimizer | ~360MB | ~60MB |
| **Total** | **~5GB** | **~800MB** |

**Result**: Fits comfortably in 8GB VRAM with batch size 4.

### 9.2 Speed

| Metric | Current Model | StyleAwareUNet |
|--------|--------------|----------------|
| Training step | ~500ms | ~80ms |
| Inference (50 steps) | ~15s | N/A |
| Inference (20 steps) | N/A | ~1.5s |
| Time per pack (10k steps) | ~1.5 hours | ~13 minutes |

### 9.3 Power/Thermal

13-minute training runs are much more manageable on a laptop than 1.5-hour runs.

---

## 10. Addressing Your Specific Concerns

### 10.1 "Contents and patterns change"

**Solution**: Explicit style references + cross-attention.

The model sees actual examples of the pack's style during inference. It can copy patterns like:
- "This pack draws scissors on crafting tables"
- "This pack makes ore crystals diamond-shaped"
- "This pack uses dark outlines on everything"

Cross-attention allows the model to look at the reference textures and decide "what should I draw here?" based on the content structure.

### 10.2 "Must generalize to unseen blocks"

**Solution**: Per-pack training on diverse blocks.

During training, the model sees many (vanilla, target) pairs from the same pack:
- (vanilla stone, Mythic stone)
- (vanilla dirt, Mythic dirt)
- (vanilla planks, Mythic planks)
- ...

It learns the transformation: "What does Mythic do to textures?"

At test time, given vanilla `redstone_ore` (unseen), it applies the learned Mythic transformation. The style references provide additional guidance ("Mythic's ores look like this").

### 10.3 "Sprite outlines change, transparency is a problem"

**Solution**: RGBA output + alpha-aware loss.

The model predicts 4 channels: RGB + Alpha.
- RGB channels learn colors and patterns
- Alpha channel learns outline shape
- Loss = L1(RGB) + BCE(Alpha)

During preprocessing:
- Keep alpha channel separate
- Composite on magenta for RGB training (as currently done)
- Pass alpha as additional target

During inference:
- Output RGBA
- Use alpha to composite onto background
- Handles items with changing outlines (apples, bows, etc.)

---

## 11. Summary and Recommendation

**My initial plan was wrong** in calling this a deterministic translation task. Your images prove it is a generative style transfer task requiring pattern understanding and creativity.

**However**, my core recommendations remain valid with revisions:

1. ✅ **Per-pack models**: Still correct. Each pack needs its own model to avoid style interference.

2. ❌ **Direct CNN**: Wrong. Replaced with small diffusion model.

3. ✅ **Small model**: Still correct. 5M parameters, not 30M.

4. ✅ **Fast training**: Still correct. 10-20 minutes per pack.

5. 🆕 **Style references**: New recommendation. Essential for pattern transfer.

6. 🆕 **Cross-attention**: New recommendation. Allows model to copy patterns from refs.

7. 🆕 **RGBA output**: New recommendation. Handles sprites and transparency.

**Bottom line**: Train small (~5M parameter) per-pack diffusion models with explicit style references and cross-attention. Use 10-20 timesteps. Predict RGBA for sprite support. This handles pattern changes, generalizes to new blocks, and trains in minutes on your 4060.

**Do not use Stable Diffusion** unless this approach completely fails. SD is overkill and too slow for your use case.

---

*Document version: 2.0*
*Date: 2026-05-07*
*Revision: Added generative understanding, style references, cross-attention, RGBA output*
