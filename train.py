from config import *
from tokenizer import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_model, load_model
from inference import *

print(f"{device = }")

dataset = FineTuneDataset(tokens_file=tokenized_data, sos_token=sos_token, eos_token=eos_token, pad_token=pad_token, sequence_len=sequence_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

transformer = Transformer(vocab_size=vocab_size, sequence_len=sequence_len, d_model=d_model, num_head=num_head, d_ff=d_ff, dropout=dropout, blocks=blocks, device=device).to(device)
load_model(transformer, pretrained_weights)
print("Loaded pretrained model")
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1).to(device)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=betas, eps=1e-9)
mean = lambda x: sum(x)/len(x)
model_size = sum(p.numel() for p in transformer.parameters())
print(f"{model_size = :,}")

with torch.no_grad():
    generated = eval(transformer=transformer, input_text=test_text, tokenizer=tokenizer, sos_token=sos_token, eos_token=eos_token, pad_token=pad_token, sequence_len=sequence_len, device=device)
    print(f"{generated = }")
    to_file(epoch=None, generated=generated)

for epoch in range(epochs):
    pbar = tqdm(dataloader)
    pbar.set_description(f"Epoch {epoch+1}/{epochs}")
    losses = []
    for input_tokens, output_tokens in pbar:
        optimizer.zero_grad()
        y = transformer(input_tokens.to(device), transformer.mask)
        loss = loss_fn(y.view(-1, vocab_size), output_tokens.view(-1).to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"Loss": loss.item()})
    # save_model(transformer, transformer_weights)
    # torch.save(optimizer.state_dict(), optimizer_file)
    with torch.no_grad():
        generated = eval(transformer=transformer, input_text=test_text, tokenizer=tokenizer, sos_token=sos_token, eos_token=eos_token, pad_token=pad_token, sequence_len=sequence_len, device=device)
        print(f"{generated = }")
        to_file(epoch, generated)

save_model(transformer, transformer_weights)
