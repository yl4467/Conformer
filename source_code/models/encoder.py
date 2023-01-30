import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask, Tri_sliding

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonality = x - trend
        return seasonality, trend

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attn, attention, enc_lstm, step_len, d_model, d_ff=None, dropout=0.05, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attn = attn
        self.attention = attention
        self.step = step_len
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
       
        self.lstm = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=enc_lstm)
        self.lstm1 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=enc_lstm)
        self.decomp1 = series_decomp(step_len)
        self.decomp2 = series_decomp(step_len)

    def forward(self, x, attn_mask=None):
        '''
        if self.attn == 'full':
            attn_mask = TriangularCausalMask(B, L, device=x.device)
        elif self.attn == 'long':
            attn_mask = Tri_sliding(B, L, H, device=x.device).mask
        else:
            attn_mask = TriangularCausalMask(B, L, device=x.device)
        '''
        new_x = self.attention(
            x, x, x, attn_mask
        )
        y1, hidden = self.lstm(x.permute(1, 0, 2).to(torch.float32))
        y1 = y1.permute(1, 0, 2)
        y1 = self.dropout(torch.softmax(y1, dim=-1)*x) + x
        x1 = self.dropout(new_x + y1)

        x, trend1 = self.decomp1(x1)
        y = self.norm1(x + self.dropout(new_x))
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        res, trend2 = self.decomp2(x + y)
        trend = trend1 + trend2
        y1, _ = self.lstm1(trend.permute(1, 0, 2))
        y1 = y1.permute(1, 0, 2)
        res = (res + y1)/2
        return res, hidden

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, hidden = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x, hidden = self.attn_layers[-1](x,  attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x, hidden = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x, hidden
