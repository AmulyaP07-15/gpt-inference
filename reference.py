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