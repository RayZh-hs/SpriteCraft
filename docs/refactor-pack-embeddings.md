# Refactor: Pack Embedding Architecture

## Overview

Replace the exemplar-driven style transfer mechanism with a **pack embedding system** conditioned on a source content exemplar. Each texture pack is represented by a learned latent style vector. The model learns to extract structure from a source image (typically vanilla) and re-render it in the target pack's style by conditioning on that pack's embedding.

This is a fundamental architectural shift from "copy and recolor vanilla via cross-attention" to "extract structure from source, then paint with pack style vector."

## Why This Approach

**Current system**: Uses support exemplars (vanilla texture + target pack texture pairs) to extract style via local cross-attention. The vanilla content reference acts as a hard structural prior, preventing dramatic style changes.

**New system**: Separates structure extraction from style application. A content encoder processes the source exemplar to extract structural features. A pack embedding provides style. At inference, you provide a source image of *any* block and a target pack's style vector — the model transforms it.

**Trade-offs**:
- **Pros**: Generalizes to unseen blocks, simpler data pipeline, faster inference, stronger style expression, easy new pack adaptation
- **Cons**: Requires training/fine-tuning for new packs (no zero-shot pack styles), source exemplar must be provided at inference

## Architecture

### Inputs

The model receives:
1. `noisy_target`: [B, H, W] integer tokens (the masked target to denoise)
2. `source_content`: [B, H, W] integer tokens (the source exemplar providing structure — typically vanilla)
3. `pack_id`: [B] integer indices (which texture pack style to apply)
4. `t`: [B] diffusion timestep

**No support images. No cross-attention over exemplar pairs. Style is provided solely by the pack embedding vector.**

### Embeddings

```python
class ContentEncoder(nn.Module):
    """Extracts spatial structure from a source exemplar."""
    def __init__(self, vocab_size=257, embed_dim=192, out_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 128)
        self.norm2 = nn.GroupNorm(8, 128)
    
    def forward(self, source_tokens):
        x = self.token_embedding(source_tokens.long())  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)                      # [B, C, H, W]
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        return x  # [B, out_dim, H, W]

class PackConditionedUNet(nn.Module):
    def __init__(self, num_packs, vocab_size=257, 
                 style_dim=256, embed_dim=192):
        
        # Content structure: extract from source exemplar
        self.content_encoder = ContentEncoder(vocab_size, embed_dim, style_dim)
        
        # Pack style: what aesthetic to apply?
        self.pack_embedding = nn.Embedding(num_packs, style_dim)
        
        # Time embedding: where in the diffusion process?
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, style_dim)
        )
        
        # Token embedding for the noisy target (unchanged)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Input projection: noisy tokens + content structure
        self.in_proj = nn.Conv2d(embed_dim + style_dim, 128, kernel_size=3, padding=1)
        
        # Standard UNet with FiLM conditioning
        self.enc32 = ResBlock(128, cond_dim=style_dim)
        # ... rest of UNet unchanged ...
```

### Forward Pass

```python
def forward(self, noisy_target, source_content, pack_id, t):
    # Extract structure from source exemplar
    content_features = self.content_encoder(source_content)  # [B, style_dim, H, W]
    
    # Embed the noisy input tokens
    x = self.token_embedding(noisy_target.long())  # [B, H, W, C]
    x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
    
    # Concatenate content structure with noisy input
    x = torch.cat([x, content_features], dim=1)  # [B, embed_dim + style_dim, H, W]
    
    # Pack style (aesthetic knowledge)
    pack_vec = self.pack_embedding(pack_id)  # [B, style_dim]
    
    # Time conditioning
    time_vec = self.time_embedding(t)  # [B, style_dim]
    
    # Combined conditioning for FiLM
    style_cond = time_vec + pack_vec  # [B, style_dim]
    
    # UNet with FiLM conditioning
    x = self.in_proj(x)
    skip32 = self.enc32(x, style_cond)
    # ... standard UNet forward ...
    
    logits = self.out(x)  # [B, vocab_size-1, H, W]
    return logits
```

### Content Structure vs. Pack Style

The architecture enforces a clean separation:

- **Content encoder** learns to extract "what this texture structurally looks like" (edges, patterns, layout) from a source exemplar
- **Pack embedding** learns "the aesthetic of the Mythic pack" (dark palette, high contrast, ornate details)
- **The UNet** learns how to combine them: structure channels guide the spatial layout, FiLM modulation from the pack vector controls the aesthetic

This is analogous to neural style transfer, but discrete, pixel-art optimized, and with pack-specific learned style vectors rather than per-image Gram matrices.

## Data Pipeline Changes

### Preprocessing

**Remove**: Support ranking, support selection, support exemplar storage, support pair indexing

**Keep**: Palette quantization, content-target pair indexing

**Add**: Source-target pair organization (vanilla source + pack target)

```python
# Episode structure
episode = {
    "source": vanilla_tokens,       # Vanilla exemplar
    "target": target_tokens,        # Target pack output
    "target_pack": "mythic",        # Which pack
    "filename": "stone.png",        # Which block type
}
```

### Dataset

The `TextureDataset` returns source-target pairs:

```python
class TextureDataset(Dataset):
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return {
            "source": episode["source"],              # Content exemplar
            "pack_id": pack_index[episode["target_pack"]],  # Style to apply
            "target": episode["target"],              # Ground truth
        }
```

**No support tensors. No support masks. No support filenames.**

### Training Loop

Simplified — no support handling:

```python
for batch in train_loader:
    source = batch["source"].to(device)
    pack_id = batch["pack_id"].to(device)
    target = batch["target"].to(device)
    
    t = torch.randint(1, NUM_TIMESTEPS + 1, (target.shape[0],), device=device)
    noisy_target = apply_mask(target, t)
    
    logits = model(noisy_target, source, pack_id, t)
    loss = F.cross_entropy(logits, target)
    
    loss.backward()
    optimizer.step()
```

**Removed**: Content preservation loss, perceptual loss, support dropout, CFG training logic

**Kept**: Cross-entropy loss (primary), time embedding, masking schedule

## Generalization to Unseen Content

**The model can generalize to blocks it has never seen during training**, provided it receives a source exemplar.

During training, the model learns that the content encoder's output should be preserved structurally while the pack vector changes colors, contrast, and texture details. If the model sees enough variety in block structures, the content encoder learns to extract generic structural features (edges, corners, patterns) rather than memorizing specific blocks.

**To verify this actually works**: Hold out specific (vanilla, pack) pairs during training. At test time, feed the held-out vanilla source and the pack ID. If the output matches the held-out target, the model has genuinely learned style transfer rather than memorization.

**Limitation**: The model cannot invent entirely new structural primitives. If the source shows a block structure unlike anything in training (e.g., a 3D model when trained on 2D flats), generalization may fail.

## Adding New Packs (Textual Inversion Style)

This is the key advantage of pack embeddings. When a new texture pack arrives:

### Step 1: Preprocess

Process the new pack through the existing pipeline:
- Extract textures
- Quantize to the shared palette (must use the SAME palette as training)
- Pair each texture with its vanilla source counterpart

### Step 2: Initialize Pack Embedding

```python
# Add a new embedding vector to the pack embedding table
new_pack_name = "external_mythic_v2"
new_pack_idx = model.pack_embedding.num_embeddings

# Initialize with mean of existing pack embeddings (warm start)
existing_mean = model.pack_embedding.weight.data.mean(dim=0)
new_embedding = nn.Parameter(existing_mean.clone().unsqueeze(0))

model.pack_embedding = nn.Embedding.from_pretrained(
    torch.cat([model.pack_embedding.weight.data, new_embedding]),
    freeze=False
)
```

### Step 3: Freeze Model, Train Embedding

```python
# Freeze everything except the new pack embedding
for param in model.parameters():
    param.requires_grad = False
model.pack_embedding.weight.requires_grad = True

# Only optimize the new pack's vector
optimizer = torch.optim.Adam([model.pack_embedding.weight[-1]], lr=1e-3)

# Train on new pack data
for epoch in range(finetune_epochs):  # 50-200 epochs is usually enough
    for source, target in new_pack_batches:
        noisy_target = apply_mask(target, t)
        pack_id = torch.full((source.shape[0],), fill_value=new_pack_idx, device=device)
        
        logits = model(noisy_target, source, pack_id, t)
        loss = F.cross_entropy(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Why this works**: The model already knows:
- How to extract structure from any source exemplar
- How to apply style vectors via FiLM
- The shared palette and token vocabulary

The new pack only needs to teach the model: "My style lives at this point in the embedding space." This typically converges in minutes, not hours.

### Step 4: Inference

```python
# Generate held-out or new block in the new pack's style
source = vanilla_tokens["ruby_ore.png"]  # Any source exemplar
pack_id = new_pack_idx

noisy_target = torch.full((1, H, W), fill_value=MASK_TOKEN)
logits = model(noisy_target, source, pack_id, t)
```

## Hierarchical Pack Embeddings (Deferred)

A single style vector per pack assumes the pack applies a content-agnostic aesthetic. In practice, some resource packs stylize different materials differently (e.g., wood vs. ore vs. stone may have distinct palette shifts).

If single-vector packs prove insufficient, the next step is a **hierarchical pack embedding**:
- One global vector per pack
- A small set of per-material basis vectors (wood, stone, metal, organic)
- At inference, combine: `style = global_vec + material_coeff * material_vec`

This is deferred until we determine whether single vectors fail in practice.

## Model Layer Changes Summary

| Layer | Current | New | Notes |
|-------|---------|-----|-------|
| `token_embedding` | nn.Embedding(257, 192) | nn.Embedding(257, 192) | Unchanged |
| `content_encoder` | N/A | ContentEncoder(257→256) | **New** — CNN extracts structure from source exemplar |
| `content_embedding` | N/A | nn.Embedding(num_contents, 128) | **Removed** — replaced by content_encoder |
| `pack_embedding` | N/A | nn.Embedding(num_packs, 256) | **New** — replaces support images |
| `time_mlp` | Linear(192, 768) → Linear(768, 192) | Linear(192, 768) → Linear(768, 256) | Output dim matches style_dim |
| `support_pair_proj` | Conv2d(384, 192) | **Removed** | No support pairs |
| `support_aggregator` | SupportAggregator(192, 8) | **Removed** | No cross-attention |
| `support_cond_proj` | Linear(192, 192) | **Removed** | No support summary |
| `in_proj` | Conv2d(576, 128) | Conv2d(448, 128) | Noisy tokens + content features |
| `ResBlock.cond_dim` | 192 | 256 | Style conditioning dimension |
| `forward()` args | noisy_target, content_ref, support_content, support_style, support_mask, t | noisy_target, source_content, pack_id, t | Simplified |

## Training Strategy

### Phase 1: Main Training

Train the full model on all known packs:

```python
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100_000)

for step in range(100_000):
    # Standard masked diffusion training with source exemplars
    # Model learns content encoder and pack embeddings jointly
```

### Phase 2: New Pack Adaptation

For each new pack, freeze model and train only its embedding:

```python
# Freeze
for param in model.parameters():
    param.requires_grad = False

# Add and optimize new embedding
new_idx = add_pack_embedding(model, "new_pack")
optimizer = Adam([model.pack_embedding.weight[new_idx]], lr=1e-3)

for epoch in range(100):
    for batch in new_pack_data:
        loss = compute_loss(model, batch, new_idx)
        loss.backward()
        optimizer.step()
```

## Risks and Mitigations

### Risk 1: Content Encoder Memorization

The content encoder might learn to ignore the source exemplar and memorize content-specific shortcuts.

**Mitigation**: During training, occasionally replace the source exemplar with a different pack's version of the same block (5% of batches). Force the encoder to extract style-invariant structure.

### Risk 2: Content-Style Entanglement

The content encoder might leak style information from the source exemplar (e.g., copying vanilla colors into the output).

**Mitigation**: Train with source exemplars from diverse packs, not just vanilla. The encoder must learn that color and texture details come from the pack vector, not the source.

### Risk 3: Palette Limitations

If a new pack uses colors outside the training palette, the model cannot represent them.

**Mitigation**: Increase palette size from 256 to 512 or 1024. The palette must be built from ALL training packs and frozen during new pack adaptation.

### Risk 4: Single Vector Insufficiency

Some packs may require different style vectors for different material types.

**Mitigation**: See "Hierarchical Pack Embeddings (Deferred)" above.

## Validation

After refactoring, validate that:

1. **Hold-out pair reconstruction** — Hide 10% of (vanilla, pack) pairs from training. The model must reconstruct them at test time given the vanilla source and pack ID. This is the primary test of true style transfer.
2. **Unseen block generalization** — Train without ever seeing "redstone_ore" in Pack A. At test time, feed vanilla redstone ore + Pack A vector. Output should be plausible Pack A redstone ore.
3. **Outputs vary significantly across packs** — brick in Mythic vs. brick in Chibli should look dramatically different, not just recolored
4. **New pack adaptation converges quickly** — < 30 minutes on a single GPU
5. **Content identity is preserved** — brick outputs should still look like bricks (recognizable pattern) even in wildly different styles
6. **No vanilla bias** — outputs should not default to vanilla-like colors or patterns when generating for non-vanilla packs

## Summary

This refactoring replaces the exemplar-driven, vanilla-templated generation pipeline with a clean separation of structure and style:

- **Content encoder** extracts structure from any source exemplar
- **Pack embeddings** learn each pack's aesthetic as a single latent vector
- **The UNet** learns to paint the extracted structure with the target style via FiLM
- **New packs** are added by learning a single embedding vector (textual inversion)
- **New blocks** are handled by providing their vanilla source exemplar at inference

The result should be true style transfer generalization, simpler code, faster inference, and extensibility to arbitrary new packs through lightweight fine-tuning.
