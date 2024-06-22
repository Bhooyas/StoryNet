import torch
import torch.nn as nn
import math

class MaskMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_head):
        super(MaskMultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        self.Q_w = nn.Linear(d_model, d_model)
        self.K_w = nn.Linear(d_model, d_model)
        self.V_w = nn.Linear(d_model, d_model)

    def forward(self, Q_x, K_x, V_x, mask=None):
        Q_batch_size, Q_sequence_len, d_model = Q_x.shape
        K_batch_size, K_sequence_len, d_model = K_x.shape
        V_batch_size, V_sequence_len, d_model = V_x.shape
        # Get Q, K and V then reshape (Batch, Sequence, D_model) -> (Batch, Heads, Sequence, D_k)
        Q = self.Q_w(Q_x).view(Q_batch_size, Q_sequence_len, self.num_head, self.d_k).transpose(1, 2)
        K = self.K_w(K_x).view(K_batch_size, K_sequence_len, self.num_head, self.d_k).transpose(1, 2)
        V = self.V_w(V_x).view(V_batch_size, V_sequence_len, self.num_head, self.d_k).transpose(1, 2)

        # Q . K
        y = Q @ K.transpose(2, 3)

        # Scale
        y = y / math.sqrt(self.d_k)

        # Apply Mask
        if mask is not None:
            y = y.masked_fill(mask == 0, -1e9)

        # Softmax
        y = torch.softmax(y, dim=3)

        # Attention
        y = y @ V

        # Reshape to (Batch, Sequence, D_model)
        y = y.transpose(1, 2).contiguous().view(Q_batch_size, Q_sequence_len, d_model)
        return y

class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):

    def __init__(self, sequence_len, d_model, dropout):
        super(PositionalEmbedding, self).__init__()

        self.sequence_len = sequence_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        positionalembedding = torch.zeros(sequence_len, d_model)

        for pos in range(sequence_len):
            for i in range(0, d_model, 2):
                positionalembedding[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                positionalembedding[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))

        positionalembedding = positionalembedding.view(1, sequence_len, d_model)
        self.register_buffer('positionalembedding', positionalembedding)

    def forward(self, x):
        x = x + self.positionalembedding[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.l1 = nn.Linear(self.d_model, self.d_ff)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        x = self.relu(self.l1(x))
        return self.l2(x)

class Block(nn.Module):

    def __init__(self, d_model, num_head, d_ff, dropout):
        super(Block, self).__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff

        self.MHA = MaskMultiHeadAttention(d_model, num_head)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention
        y = self.dropout(self.MHA(x, x, x, mask))
        # Connection
        x = self.norm1(x + y)
        # FeedForward
        y = self.dropout(self.ff(x))
        # Connection
        x = self.norm2(x + y)
        return x
