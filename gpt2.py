import torch


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, correction=0)

    normalized = (x - mean) / torch.sqrt(variance + eps)
    return normalized * weight + bias

@torch.no_grad()
def transformer_block(x, block):
    batch_size, num_tokens, hidden_size = x.shape

    # First LayerNorm.
    ln = block.ln_1
    normalized_x = layer_norm(x, ln.weight, ln.bias, ln.eps)

    # Create Q, K and V.
    projection = block.attn.c_attn
    qkv = normalized_x @ projection.weight + projection.bias
    q, k, v = qkv.split(hidden_size, dim=-1)

    # Arrange the 12 heads.
    q = q.reshape(batch_size, num_tokens, 12, 64).transpose(1, 2)
    k = k.reshape(batch_size, num_tokens, 12, 64).transpose(1, 2)
    v = v.reshape(batch_size, num_tokens, 12, 64).transpose(1, 2)

    # Calculate and scale attention scores.
    scores = q @ k.transpose(-2, -1)
    scores = scores / (q.shape[-1] ** 0.5)

    # Block future tokens.
    mask = torch.ones(num_tokens, num_tokens, device=x.device)
    mask = torch.tril(mask).bool()
    scores = scores.masked_fill(~mask, float("-inf"))

    # Blend value vectors and merge heads.
    attention_weights = torch.softmax(scores, dim=-1)
    head_output = attention_weights @ v
    merged_heads = head_output.transpose(1, 2)
    merged_heads = merged_heads.reshape(batch_size, num_tokens, hidden_size)

    # Attention projection and first residual.
    output_projection = block.attn.c_proj
    attention_output = (
        merged_heads @ output_projection.weight + output_projection.bias
    )
    x = x + attention_output

    # Second LayerNorm.
    ln2 = block.ln_2
    normalized_x = layer_norm(x, ln2.weight, ln2.bias, ln2.eps)

    # MLP expansion, GELU and projection.
    fc = block.mlp.c_fc
    expanded = normalized_x @ fc.weight + fc.bias
    activated = torch.nn.functional.gelu(expanded, approximate="tanh")

    mlp_projection = block.mlp.c_proj
    mlp_output = activated @ mlp_projection.weight + mlp_projection.bias

    # Second residual.
    x = x + mlp_output
    return x

@torch.no_grad()
def gpt2_forward(input_ids, model):
    # Token + position embeddings: [1, 4] → [1, 4, 768].
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    x = model.transformer.wte(input_ids) + model.transformer.wpe(positions)

    # Run all 12 blocks in order, each with its own weights.
    # Each block's output becomes the next block's input.
    for block in model.transformer.h:
        x = transformer_block(x, block)

    # Final LayerNorm after the last block.
    ln_f = model.transformer.ln_f
    x = layer_norm(x, ln_f.weight, ln_f.bias, ln_f.eps)

    # Score every vocabulary token: [1, 4, 768] @ [768, 50257] → [1, 4, 50257].
    # GPT-2 reuses the token embedding table here (weight tying).
    logits = x @ model.transformer.wte.weight.T
    return logits


