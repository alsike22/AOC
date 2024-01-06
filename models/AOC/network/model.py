from torch import nn
import torch


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.project_channels = configs.project_channels
        self.hidden_size = configs.hidden_size
        self.window_size = configs.window_size
        self.device = device
        self.num_layers = configs.num_layers
        self.dropout = configs.dropout

        self.encoder = nn.LSTM(
            self.input_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )
        self.decoder = nn.LSTM(
            self.input_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )

        self.output_layer = nn.Linear(self.hidden_size, self.input_channels)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.project_channels, bias=False),
        )

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x, return_latent=True, training=True):
        enc_hidden = self.init_hidden_state(x.shape[0])
        _, enc_hidden = self.encoder(x.float(), enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        hidden = self.projection_head(enc_hidden[1][-1])
        return (output, hidden) if return_latent else output


