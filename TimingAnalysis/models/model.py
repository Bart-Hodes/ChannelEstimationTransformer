import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(RNNUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
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
        output, cur_hidden = self.rnn(output, prev_hidden)
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class RNN(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(
            features, input_size, hidden_size, num_layers=self.num_layers
        )

    def train_data(self, x, device):

        BATCH_SIZE, seq_len, _ = x.shape
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


class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(LSTMUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden, prev_cell):
        # if len(x.shape) > 3:
        # print('x shape must be 3')

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        # print(1)
        output = output.reshape(L, B, -1)
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        # print(2)
        output = output.reshape(L * B, -1)
        # print(3)
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden, cur_cell


class LSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(
            features, input_size, hidden_size, num_layers=self.num_layers
        )

    def train_data(self, x, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(
            device
        )
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(
            device
        )
        outputs = [x[:, 0:1, ...].permute(1, 0, 2).contiguous()]
        for idx in range(seq_len - 1):
            output, prev_hidden, prev_cell = self.model(
                x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous(),
                prev_hidden,
                prev_cell,
            )
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def test_data(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(
            device
        )
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(
            device
        )

        outputs = []
        outputs_debug = [x[:, 0:1, ...].permute(1, 0, 2).contiguous()]
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(
                    x[:, idx : idx + 1, ...].permute(1, 0, 2).contiguous(),
                    prev_hidden,
                    prev_cell,
                )
                # print("Train: ", idx)
            else:
                output, prev_hidden, prev_cell = self.model(
                    output, prev_hidden, prev_cell
                )

            if idx >= seq_len - 1:
                outputs.append(output)
            outputs_debug.append(output)

        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()
        outputs_debug = torch.cat(outputs_debug, dim=0).permute(1, 0, 2).contiguous()

        return outputs, outputs_debug
