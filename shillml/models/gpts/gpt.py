import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from tqdm import tqdm


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers, n_heads, n_embd, dropout=0.1):
        super(GPT, self).__init__()
        self.block_size = block_size
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, target=None):
        b, t = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.token_embedding(x)
        pos_embeddings = self.pos_embedding(torch.arange(t, device=x.device))
        x = token_embeddings + pos_embeddings

        x = self.transformer(x, x)
        x = self.ln_f(x)
        logits = self.head(x)

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            return logits, loss
        return logits


class TextDataset(Dataset):
    def __init__(self, texts, block_size, vocab):
        self.block_size = block_size
        self.vocab = vocab
        self.data = self.process_texts(texts)

    def process_texts(self, texts):
        data = ''.join(texts)
        return data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        input_ids = torch.tensor([self.vocab[ch] for ch in chunk[:-1]], dtype=torch.long)
        target_ids = torch.tensor([self.vocab[ch] for ch in chunk[1:]], dtype=torch.long)
        return input_ids, target_ids

def build_vocab(texts):
    unique_chars = sorted(list(set(''.join(texts))))
    vocab = {ch: i for i, ch in enumerate(unique_chars)}
    return vocab

def train(model, dataset, epochs, batch_size, learning_rate, device):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for input_ids, target_ids in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            logits, loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}')

    return model

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, target_ids in data_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits, loss = model(input_ids, target_ids)
            total_loss += loss.item()
    return total_loss / len(data_loader)


if __name__ == "__main__":
    # hyperparameters (tune these as needed)
    vocab_size = 128  # size of the vocabulary
    block_size = 128  # context window size
    n_layers = 6  # number of transformer layers
    n_heads = 8  # number of attention heads
    n_embd = 256  # embedding size
    dropout = 0.1  # dropout rate
    epochs = 10  # number of epochs
    batch_size = 64  # batch size
    learning_rate = 1e-4  # learning rate

    # load dataset
    dataset = load_dataset("wikitext", 'wikitext-103-raw-v1',split="train[:1%]")  # load a small portion of the dataset
    texts = dataset['text']

    # build vocabulary
    vocab = build_vocab(texts)

    # create dataset
    text_dataset = TextDataset(texts, block_size, vocab)

    # instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(len(vocab), block_size, n_layers, n_heads, n_embd, dropout).to(device)

    # train the model
    model = train(model, text_dataset, epochs, batch_size, learning_rate, device)

    # save the model state dict
    torch.save(model.state_dict(), "gpt.pth")
