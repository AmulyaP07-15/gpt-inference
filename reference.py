import torch
from transformers import AutoTokenizer,AutoModelForCausalLM



#STEP1: ENCODE-DECODE using the same tokenizer 
tokenizer=AutoTokenizer.from_pretrained("gpt2")


model=AutoModelForCausalLM.from_pretrained("gpt2")
#to disbale the traning behaviour
model.eval()

prompt="The quick brown fox"

#return_tensors="pt" return the output as a tensor 
input_ids=tokenizer.encode(prompt,return_tensors="pt")

#the projection.weight  combines w_q,w_k,w_v
projection=model.transformer.h[0].attn.c_attn

print("Token IDs:", input_ids)
print("Shape :",input_ids.shape)


for token_id in input_ids[0].tolist():
    print(token_id,
    
    repr(tokenizer.decode([token_id])))

#wte is the one learned 768 vector for each of the 50,257 token IDs
print("Token embedding table:",
model.transformer.wte.weight.shape)

#wpe is the learned 768 number vector for each of the 1024 supported positions
print("Position embedding table:",
model.transformer.wpe.weight.shape)

#look up the token embeddings+position embeddings
with torch.no_grad():
    token_embeddings=model.transformer.wte(input_ids)
    positions=torch.arange(input_ids.shape[1])

    position_embeddings=model.transformer.wpe(positions)
    x= token_embeddings+position_embeddings

#layernormalization
ln=model.transformer.h[0].ln_1

#to compare the layer norm func with pytorch uilt in function
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, correction=0)

    normalized = (x - mean) / torch.sqrt(variance + eps)
    return normalized * weight + bias


with torch.no_grad():
    normalized_x=layer_norm(x,ln.weight,ln.bias,ln.eps)

with torch.no_grad():
    expected=ln(x)
    difference=(normalized_x-expected).abs().max()

print("Maximum difference",difference)

#qkv
#[1,4,768] @ [768,2304]-> [1,4,2304]
#Q: 768 | K: 768 | V: 768 
#@ means matrix multiplication
qkv=normalized_x @ projection.weight+projection.bias

q,k,v=qkv.split(768,dim=1)


print("Q",q.shape)
print("K",k.shape)
print("V",v.shape)

#each currently has the shape of [1,4,768] and the 768 has to be split across 12 heads
#q.shape->[1,4,768]->[1,4,12,64]->after transpose[1,12,4,64]

batch_size,num_tokens,hidden_size=q.shape


q=q.reshape(batch_size,num_tokens,12,64).transpose(1,2)
k=k.reshape(batch_size,num_tokens,12,64).transpose(1,2)
v=v.reshape(batch_size,num_tokens,12,64).transpose(1,2)

print("Q heads:", q.shape)

#now calculating the attention

with torch.no_grad():
    scores=q @ k.transpose(-2,-1)
    scores=scores/(q.shape[-1]**0.5)

print("scores shape:", scores.shape)
#scores[0,0] the first batch and the first head 
print("First head: ", scores[0,0])


num_tokens=scores.shape[-1]
#num of tokens is 4

#creates a separate matrix of ones
mask=torch.ones(num_tokens,num_tokens,device=scores.device)

#keep the lower triangle
mask=torch.tril(mask)

#change the numbers to bool
mask=mask.bool()

# ~mask selects blocked positions, which correspond to future tokens.
# Replace only those scores with negative infinity.
# Keep allowed scores unchanged. Apply the mask to every batch and head.
# Softmax will later turn the blocked scores into zero attention weights.
scores=scores.masked_fill(~mask,float("-inf"))

# q=[4,64] @ [64,4] after the transpose
#after the q @k.T multiplication the tensors transform into [1,12,4,64] to [1,12,4,4] 
# the softmax raises each score[i,j] to e^score[i,j] and adds them and divides of each of them by total 
# softmax returns each of the numbers between 0 and 1 ,which sums up to 1 per row  
attention_weights=torch.softmax[scores,dim=-1]

#now the output
with torch.no_grad():
    head_output=attention_weights @ v 

print("Head output shape" , head_output.shape)

#for one head its going to attention_weights=[4,4] @ v vectors[4,64] ->output[4,64]


#now merge all the heads
#before that transpose[1,12,4,64]->[1,4,12,64]
merged_heads=head_output.transpose(1,2)

merged_heads=merged_heads.reshape(batch_size,num_tokens,768)


print("Merged shape:", merged_heads.shape)



