from layers import *

class Transformer(nn.Module):

    def __init__(self, vocab_size=5000, sequence_len=800, d_model=512, num_head=8, d_ff=2048, dropout=0.1, blocks=6, device="cpu"):
        super(Transformer, self).__init__()

        self.embedding = Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(sequence_len, d_model, dropout)

        self.layers = nn.ModuleList([Block(d_model, num_head, d_ff, dropout) for _ in range(blocks)])

        self.l1 = nn.Linear(d_model, vocab_size)

        self.mask = (1 - torch.triu(torch.ones(1, sequence_len, sequence_len), diagonal=1)).to(device)

    def forward(self, x, mask=None):
        # Generate input embeddings
        embeddings = self.positional_embedding(self.embedding(x))

        # Pass it through layers
        output = embeddings
        for layer in self.layers:
            output = layer(output, mask)

        # Get the predicted tokens
        y = self.l1(output)
        return y
