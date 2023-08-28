import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import math

#hypeparameters
batch_size = 4
block_size = 8
learning_rate = 0.001
iterations = 50000
num_eval = 1000
num_emb = 32
head_size = 16

#read data
with open('sheakspeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #all dymbols in text
vocab_size = len(chars)

# symbol to index
stoi = {s:i for i,s in enumerate(chars)}
# index to symbol
itos = {i:s for i,s in enumerate(chars)}
# encode a string
encode = lambda s: [stoi[c] for c in s]
# decode an array
decode = lambda a: ''.join([itos[i] for i in a])

# encode entire text
data = torch.tensor(encode(text), dtype=torch.long)

# train test split of data
train = data[:int(len(data) * 0.9)]
test = data[int(len(data) * 0.9):]
len(train), len(test)

def get_batch(data_type):
    if data_type == 'train':
        data = train
    elif data_type == 'test':
        data = test
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

#estimate average loss to reduce noise
@torch.no_grad()
def estiamte_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        loss_tensor = torch.zeros(num_eval)
        for i in range(num_eval):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            loss_tensor[i] = loss
            out[split] = loss_tensor.mean()
    model.train()
    return out

#model itself
class Attention_head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(num_emb, head_size, bias=False)
        self.key = nn.Linear(num_emb, head_size, bias=False)
        self.value = nn.Linear(num_emb, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        weights = query @ key.transpose(1, 2) / math.sqrt(num_emb)
        weights = weights.masked_fill(self.mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        return weights @ value
    
class Multi_head_attention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Attention_head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_emb, num_emb)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, num_emb):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_emb, 4 * num_emb),
            nn.ReLU(),
            nn.Linear(4 * num_emb, num_emb),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class Transformer_block(nn.Module):
    def __init__(self, num_emb, num_heads):
        super().__init__()
        head_size = num_emb // num_heads
        self.attention = Multi_head_attention(num_heads, head_size)
        self.feed_forward = FeedForward(num_emb)
        self.layernorm1 = nn.LayerNorm(num_emb)
        self.layernorm2 = nn.LayerNorm(num_emb)
    
    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feed_forward(self.layernorm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_emb)
        self.position_embedding = nn.Embedding(block_size, num_emb)
        self.blocks = nn.Sequential(
            Transformer_block(num_emb, 4),
            Transformer_block(num_emb, 4),
            Transformer_block(num_emb, 4),
        )
        self.lang_model_head = nn.Linear(num_emb, vocab_size)


    def forward(self, x, y=None): #(B, T) tensor
        token_emb = self.token_embedding(x) #(B, T, C) tensor
        pos_emb = self.position_embedding(torch.arange(x.shape[1])) #(T, C) 
        emb = token_emb + pos_emb #(B, T, C)
        x = self.blocks(emb) #(B, T, C)
        logits = self.lang_model_head(x) #(B, T, vocab_size) 
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, x, max_tokens):
        for i in range(max_tokens):
            x_curr = x[:, -block_size:]
            logits, loss = self(x_curr)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, 1)
            x = torch.cat((x, x_next), dim=1)
        return x

    
model = Transformer()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#train model
for i in range(iterations):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
out = estiamte_loss()
print(out)

