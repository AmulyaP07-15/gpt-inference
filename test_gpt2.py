import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gpt2 import layer_norm, transformer_block, gpt2_forward


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    # Hugging Face's reference outputs.
    reference = model(
        input_ids,
        output_hidden_states=True,
        use_cache=False,
    )

    # Create our starting embeddings.
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    x = model.transformer.wte(input_ids) + model.transformer.wpe(positions)

    # Compare outputs as we run through the blocks.
    last_block = len(model.transformer.h) - 1

    for i, block in enumerate(model.transformer.h):
        x = transformer_block(x, block)

        if i < last_block:
            ours = x
            expected = reference.hidden_states[i + 1]
            label = f"Block {i}"
        else:
            # HF applies final LayerNorm before saving its last hidden state.
            ln_f = model.transformer.ln_f
            ours = layer_norm(x, ln_f.weight, ln_f.bias, ln_f.eps)
            expected = reference.hidden_states[-1]
            label = f"Block {i} + final LayerNorm"

        difference = (ours - expected).abs().max().item()
        print(f"{label}: maximum difference = {difference:.6e}")

    # Test our complete forward function, including vocabulary projection.
    our_logits = gpt2_forward(input_ids, model)
    logit_difference = (our_logits - reference.logits).abs().max().item()

    # Pick the next token from the last position's vocabulary scores.
    our_next = our_logits[0, -1].argmax().item()
    expected_next = reference.logits[0, -1].argmax().item()

print()
print("Logits shape:", our_logits.shape)
print("Logits maximum difference:", logit_difference)
print("Logits within 1e-4:", logit_difference < 1e-4)
print("Our next token:", our_next, repr(tokenizer.decode([our_next])))
print("HF next token: ", expected_next, repr(tokenizer.decode([expected_next])))
print("Next tokens match:", our_next == expected_next)
