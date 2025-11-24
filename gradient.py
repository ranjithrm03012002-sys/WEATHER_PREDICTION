import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

##############################################################
# 1 DATA LOADING & PREPROCESSING
##############################################################

# Replace with real dataframe load. Example format (df: [timestamp, feat1..featN])
df = pd.read_csv("energy.csv", parse_dates=['timestamp'], index_col='timestamp')
df = df.fillna(method="ffill")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)

input_seq_len = 48       # past timesteps used for encoder
output_seq_len = 24      # future timesteps predicted
features = scaled_data.shape[1]


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_len, output_len):
        self.data = data
        self.in_len = input_len
        self.out_len = output_len

    def __len__(self):
        return len(self.data) - self.in_len - self.out_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.in_len]
        y = self.data[idx+self.in_len:idx+self.in_len+self.out_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

dataset = TimeSeriesDataset(scaled_data, input_seq_len, output_seq_len)
train_size = int(len(dataset)*0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)

##############################################################
# 2 SEQ2SEQ WITH ATTENTION
##############################################################

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, hidden, enc_outputs):
        hidden = hidden[-1].unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
        energy = self.attn(torch.cat((hidden, enc_outputs), dim=2))
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim):
        super().__init__()
        self.attention = Attention(hid_dim)
        self.lstm = nn.LSTM(hid_dim+output_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, y_prev, hidden, cell, enc_outputs):
        context, weights = self.attention(hidden, enc_outputs)
        rnn_input = torch.cat((y_prev, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt_len):
        enc_outputs, hidden, cell = self.encoder(src)
        outputs = []
        attn_weights = []
        y_prev = src[:, -1, :].unsqueeze(1)

        for _ in range(tgt_len):
            out, hidden, cell, weights = self.decoder(y_prev, hidden, cell, enc_outputs)
            outputs.append(out.unsqueeze(1))
            attn_weights.append(weights.unsqueeze(1))
            y_prev = out.unsqueeze(1)

        return torch.cat(outputs, dim=1), torch.cat(attn_weights, dim=1)


hid_dim = 128
encoder = Encoder(features, hid_dim)
decoder = Decoder(features, hid_dim)
model = Seq2SeqAttention(encoder, decoder).to("cpu")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

##############################################################
# 3 TRAINING LOOP
##############################################################

def train_epoch():
    model.train()
    total_loss = 0
    for x,y in train_loader:
        optimizer.zero_grad()
        y_pred, _ = model(x, output_seq_len)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for x,y in test_loader:
            y_pred, _ = model(x, output_seq_len)
            preds.append(y_pred.numpy())
            actual.append(y.numpy())
    preds, actual = np.vstack(preds), np.vstack(actual)
    return rmse(preds, actual), mae(preds, actual)


def rmse(p,a): return np.sqrt(mean_squared_error(a.flatten(), p.flatten()))
def mae(p,a): return mean_absolute_error(a.flatten(), p.flatten())


for epoch in range(40):
    loss = train_epoch()
    if (epoch+1) % 5 == 0:
        R, M = evaluate()
        print(f"Epoch {epoch+1} | Loss={loss:.4f} | RMSE={R:.4f} | MAE={M:.4f}")

##############################################################
# 4 ATTENTION WEIGHTS EXTRACTION (Interpretation)
##############################################################

with torch.no_grad():
    for x,y in test_loader:
        preds, attn = model(x, output_seq_len)
        example_attention = attn[0].numpy()
        break

np.savetxt("attention_weights.csv", example_attention, delimiter=",")


class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        outputs, (h,c) = self.lstm(x)
        output = self.fc(outputs[:, -1, :]).unsqueeze(1)
        return output.repeat(1, output_seq_len, 1)

baseline = BaselineLSTM(features, 128, features)
optimizer_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
