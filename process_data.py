from datasets import load_dataset
from config import *
from tqdm import tqdm
import pickle
import youtokentome as yttm

dataset = load_dataset("roneneldan/TinyStories")
data = ""
story = []

pbar = tqdm(dataset["train"])
for i in pbar:
    temp = len(i["text"].split())
    if temp <= words:
        data += i["text"]
        story.append(i["text"])
        pbar.set_postfix({"Len": len(story)})

    if len(story) == num_stories:
        break

print(f"{len(story) = }")

with open("train_data.txt", "w", encoding='utf-8') as file:
    file.write(data)

yttm.BPE.train(data="train_data.txt", vocab_size=vocab_size, model=tokenizer_file)
tokenizer = yttm.BPE(model=tokenizer_file)

tokens = []
story_tokens = []
story_len = []

for i in story:
        temp = tokenizer.encode(i)
        tokens.extend(temp)
        story_tokens.append(temp)
        story_len.append(len(temp))

with open(combined_data, "wb") as file:
    pickle.dump(tokens, file)

with open(tokenized_data, "wb") as file:
    pickle.dump(story_tokens, file)

print(f"{min(story_len) = } {max(story_len) = } {sum(story_len)/len(story_len) = }")
