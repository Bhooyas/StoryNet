import torch
import youtokentome as yttm

# files config
tokenizer_file = "./tokenizer.yttm" # Name of the tokenizer file
combined_data = "combined_data.pkl" # Name of the file where combined sotries are stored for pretraining
tokenized_data = "tokenized_stories.pkl" # Name of the file where stories are stored indiviually

# Process data
num_stories = 1_00_000 # Number of stories to use for training
words = 250 # Maximum words a story can have

# Tokenizer
try:
    tokenizer = yttm.BPE(model=tokenizer_file) # Loading the tokenizer
    sos_token = tokenizer.subword_to_id("<BOS>") # Get the <BOS> token id
    eos_token = tokenizer.subword_to_id("<EOS>") # Get the <EOS> token id
    pad_token = tokenizer.subword_to_id("<PAD>") # Get the <PAD> token id
except:
    print("Cannot load the tokenizer please run process_data.py")


# Transformer config
sequence_len = 350 # Sequence Len
vocab_size = 5000 # Vocab Size for tokenizer
d_model = 256 # Dmodel
num_head = 8 # Number of heads in attention
d_ff = d_model * 4 # Feedforward size
dropout = 0.1 # Dropout
blocks = 2 # Number of decoder Blocks
lr = 1e-4 # Lr for Optimizer
betas = (0.9, 0.98) # Betas for optimizers
device = "cuda" if torch.cuda.is_available() else "cpu" # Device to Train on

# pretrain config
pretrained_weights = "transformer_pretrained.safetensors" # The file to store pretrained model weights
pre_epochs = 1 # Epochs for pretraining
pre_lr = 1e-4 # Pretraining lr
pre_test = "Once" # String to test for inference while pretraining

# train config
transformer_weights = "transformer.safetensors" # The file to store the model weights
batch_size = 64 # Batch size for training
epochs = 50 # Epochs of taining
test_text = "Once" # String to test for inference
