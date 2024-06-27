import torch
import torch.nn as nn


class GRUUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(GRUUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.gru(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class GRU(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(
            features, input_size, hidden_size, num_layers=self.num_layers
        )

    def train_data(self, x, device):

        BATCH_SIZE, seq_len, _ = x.shape
        x = x.to(device)
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(
            device
        )
        outputs = [x[:, 0:1, ...].permute(1, 0, 2).contiguous()]
        for idx in range(seq_len - 1):
            output, prev_hidden = self.model(
                x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden
            )
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def test_data(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(
            device
        )
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(
                    x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden
                )
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x):
        device = x.device
        return self.train_data(x, device)
