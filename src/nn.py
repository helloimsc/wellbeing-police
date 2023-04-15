import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(
        self, inputs, labels, tokenizer, vocab_encoder, max_length, impose_len=False
    ):
        self.inputs = [vocab_encoder(tokenizer(x)) for x in inputs]
        self.inputs = [torch.tensor(x[:max_length]) for x in self.inputs]
        if impose_len:
            self.inputs.append(torch.empty(max_length))
        self.inputs = pad_sequence(self.inputs, batch_first=True, padding_value=0)
        if impose_len:
            self.inputs = self.inputs[:-1, :]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


class GloveDataset(Dataset):
    def __init__(self, glove, inputs, labels, tokenizer, max_length, impose_len=False):
        self.inputs = [
            glove.get_vecs_by_tokens(tokenizer(x), lower_case_backup=True)
            for x in inputs
        ]
        self.inputs = [x[:max_length].view(-1) for x in self.inputs]

        if impose_len:
            self.inputs.append(torch.empty(max_length * 300))

        self.inputs = pad_sequence(self.inputs, batch_first=True, padding_value=0)

        if impose_len:
            self.inputs = self.inputs.view(len(inputs) + 1, -1, 300)
            self.inputs = self.inputs[:-1, :, :]
        else:
            self.inputs = self.inputs.view(len(inputs), -1, 300)

        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


# Did not submit as easily overfits with trainable embeddings
class RNNClassifier(nn.Module):
    def __init__(
        self, input_size, output_size, embedding_size, hidden_size, num_layers
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(input_size, embedding_size),
            nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False,
                num_layers=num_layers,
            ),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(hidden_size * num_layers, output_size)

    def forward(self, x):
        y, (hfinal, _) = self.encoder(x)
        y = torch.transpose(hfinal, 0, 1)
        y = self.flatten(y)
        y = self.classifier(y)
        return y


class RNNGloveClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=300,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(hidden_size * num_layers, output_size)

    def forward(self, x):
        _, hfinal = self.encoder(x)
        y = torch.transpose(hfinal, 0, 1)
        y = self.flatten(y)
        y = self.classifier(y)
        return y


# Did not submit as easily overfits with trainable embeddings
class CNNClassifier(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, out_channels, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bigram = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_size,
                out_channels=out_channels,
                kernel_size=2,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(),
        )
        self.trigram = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_size,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(),
        )
        self.tetragram = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_size,
                out_channels=out_channels,
                kernel_size=4,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(),
        )
        self.agg = nn.Conv1d(out_channels * 3, 50, 1)
        self.batchnorm = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.classifier = nn.Conv1d(50, 5, 1)
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        y = self.embedding(x)
        y = torch.transpose(y, 1, 2)
        y1 = self.bigram(y)
        y2 = self.trigram(y)
        y3 = self.tetragram(y)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.agg(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.classifier(y)
        y = self.maxpool(y)
        y = self.flatten(y)

        return y


class CNNGloveClassifier(nn.Module):
    def __init__(self, output_size, seq_len):
        super().__init__()
        self.bigram = nn.Sequential(
            nn.Conv1d(
                in_channels=300,
                out_channels=60,
                kernel_size=2,
                padding="same",
            ),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(),
        )
        self.trigram = nn.Sequential(
            nn.Conv1d(
                in_channels=300,
                out_channels=60,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(),
        )
        self.tetragram = nn.Sequential(
            nn.Conv1d(
                in_channels=300,
                out_channels=60,
                kernel_size=4,
                padding="same",
            ),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout1d(),
        )
        self.agg = nn.Conv1d(180, 50, 1)
        self.batchnorm = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.classifier = nn.Conv1d(50, 5, 1)
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        y = torch.transpose(x, 1, 2)
        y1 = self.bigram(y)
        y2 = self.trigram(y)
        y3 = self.tetragram(y)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.agg(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.classifier(y)
        y = self.maxpool(y)
        y = self.flatten(y)
        return y
