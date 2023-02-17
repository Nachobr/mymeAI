import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the model architecture
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Shape of x: (batch_size, sequence_length)
        x = x.permute(1, 0)
        # Shape of x: (sequence_length, batch_size)
        embedding = self.embedding(x)
        # Shape of embedding: (sequence_length, batch_size, embedding_dim)
        output, hidden = self.rnn(embedding)
        # Shape of output: (sequence_length, batch_size, hidden_dim)
        # Shape of hidden: (1, batch_size, hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous().view(-1, hidden_dim)
        # Shape of hidden: (batch_size, hidden_dim)
        output = self.fc(hidden)
        # Shape of output: (batch_size, output_dim)
        return output

# Prepare the text for training
def prepare_sequence(text, word_to_idx):
    words = text.split()
    tokens = [word_to_idx[word] for word in words]
    return torch.tensor(tokens, dtype=torch.long)

# Create the vocabulary and mapping from words to indices
def build_vocab(texts):
    word_to_idx = {}
    for text in texts:
        for word in text.split():
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    return word_to_idx, idx_to_word

# Train the model
def train_model(model, texts, word_to_idx, criterion, optimizer, batch_size, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = [prepare_sequence(text, word_to_idx) for text in batch_texts]
            targets = [prepare_sequence(text[1:], word_to_idx) for text in batch_texts]
            inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
            model.zero_grad()
            output = model(inputs)
            output = output.view(-1, model.output_dim)
            targets = targets.view(-1)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch {}: Loss = {}".format(epoch + 1, total_loss / len(texts)))
    return model

# Pre-process the text to clean and format it
texts = [preprocess_text(text) for text in messages]

# Build the vocabulary and mapping from words to indices
word_to_idx, idx_to_word = build_vocab(texts)

# Define the model hyperparameters
vocab_size = len(word_to_idx)
embedding_dim = 128
hidden_dim = 128
output_dim = vocab_size

# Initialize the model, criterion, and optimizer
model = LanguageModel(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
batch_size = 64
num_epochs = 5
model = train_model(model, texts, word_to_idx, criterion, optimizer, batch_size, num_epochs)

