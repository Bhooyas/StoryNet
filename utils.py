import torch
from torch.utils.data import Dataset
from config import *
import pickle

class TransformerDataset(Dataset):

    def __init__(self, tokens_file, sequence_len):
        super(TransformerDataset, self).__init__()
        with open(tokens_file, "rb") as file:
            tokens = pickle.load(file)
        self.tokens = tokens
        self.sequence_len = sequence_len
        # for i in range(len(tokens)-sequence_len-1):
        # for i in range(0, len(tokens)-1, sequence_len//16):
            # if len(tokens[i:i+sequence_len]) == sequence_len:
                # self.tokens.append({"input": tokens[i:i+sequence_len], "output": tokens[i+1:i+sequence_len+1]})

    def __len__(self):
        return len(self.tokens) - self.sequence_len - 1

    def __getitem__(self, idx):
        input_tokens = self.tokens[idx:idx+self.sequence_len]
        output_tokens = self.tokens[idx+1:idx+self.sequence_len+1]
        return torch.tensor(input_tokens, dtype=torch.int64), torch.tensor(output_tokens, dtype=torch.int64)

class FineTuneDataset(Dataset):

    def __init__(self, tokens_file, sequence_len, sos_token, eos_token, pad_token):
        super(FineTuneDataset, self).__init__()
        with open(tokens_file, "rb") as file:
            self.tokens = pickle.load(file)
        self.sequence_len = sequence_len
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token


    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        pad_len = self.sequence_len - len(tokens) - 1
        if pad_len < 0:
            input_tokens = [self.sos_token] + tokens[:pad_len]
            output_tokens = tokens[:pad_len] + [self.eos_token]
        else:
            input_tokens = [self.sos_token] + tokens + [self.pad_token for i in range(pad_len)]
            output_tokens = tokens + [self.eos_token] + [self.pad_token for i in range(pad_len)]
        return torch.tensor(input_tokens, dtype=torch.int64), torch.tensor(output_tokens, dtype=torch.int64)
