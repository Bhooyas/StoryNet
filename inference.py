import torch
import torch.nn as nn
from tokenizer import *
from config import *
from model import *
from safetensors.torch import load_model

def pre_eval(transformer, input_text, tokenizer, sequence_len, device):
    tokens = tokenizer.encode(input_text)
    pad_len = sequence_len - len(tokens) - 1
    if pad_len < 0:
        return input_text
    input_tokens = tokens
    input_tokens = torch.tensor(input_tokens, dtype=torch.int64).view(1, len(input_tokens)).to(device)
    while input_tokens.size(1) < sequence_len:
        embeddings = transformer.positional_embedding(transformer.embedding(input_tokens))
        mask = (1 - torch.triu(torch.ones(1, input_tokens.size(1), input_tokens.size(1)), diagonal=1)).to(device)
        output = embeddings
        for layer in transformer.layers:
            output = layer(output, mask)
        y = transformer.l1(output[:, -1])
        _, z = torch.max(y, dim=1)
        z = z.view(1, 1)
        input_tokens = torch.cat([input_tokens, z], dim=1)
    return tokenizer.decode(input_tokens.tolist()[0])

def eval(transformer, input_text, tokenizer, sos_token, eos_token, pad_token, sequence_len, device=device):
    # Tokenize
    tokens = tokenizer.encode(input_text)
    pad_len = sequence_len - len(tokens) - 1
    if pad_len < 0:
        return input_text
    input_tokens = [sos_token] + tokens
    input_tokens = torch.tensor(input_tokens, dtype=torch.int64).view(1, len(input_tokens)).to(device)
    while input_tokens.size(1) < sequence_len:
        embeddings = transformer.positional_embedding(transformer.embedding(input_tokens))
        mask = (1 - torch.triu(torch.ones(1, input_tokens.size(1), input_tokens.size(1)), diagonal=1)).to(device)
        output = embeddings
        for layer in transformer.layers:
            output = layer(output, mask)
        y = transformer.l1(output[:, -1])
        _, z = torch.max(y, dim=1)
        z = z.view(1, 1)
        input_tokens = torch.cat([input_tokens, z], dim=1)
        if z.item() == eos_token:
            # print(f"{z.item() = }")
            break
    return tokenizer.decode(input_tokens.tolist()[0])

def to_file(epoch, generated, file_name="story.txt"):
    with open(file_name, "a", encoding="utf-8") as file:
        if epoch == None:
            file.write(f"Before training\n")
        else:
            file.write(f"\nEpoch {epoch+1}/{epochs}\n")
        file.write(f"{generated}")

if __name__ == '__main__':
    transformer = Transformer(vocab_size=vocab_size, sequence_len=sequence_len, d_model=d_model, num_head=num_head, d_ff=d_ff, dropout=dropout, blocks=blocks, device=device).to(device)
    load_model(transformer, transformer_weights)
    input_text = input("Please Enter Text to generate rhyme\n>>>")
    generated = eval(transformer=transformer, input_text=input_text, tokenizer=tokenizer, sos_token=sos_token, eos_token=eos_token, pad_token=pad_token, sequence_len=sequence_len, device=device)
    print(generated)
