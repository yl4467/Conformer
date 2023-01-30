import torch
import torch.nn as nn
import torch.nn.functional as F

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
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

def exp_moving_ave(theta, beta):
    #theta = np.array(theta).reshape((-1, 1))
    m, n, p= theta.shape
    #v = np.zeros((m, 1))
    for i in range(1, n):
        theta[:, i, :] = (beta * theta[:, i-1, :] + (1 - beta) * theta[:, i, :])
    for i in range(1, m):
        theta[:, i, :] /= (1 - beta**i)
    return theta

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, dec_lstm, step_len, d_model, c_out, d_ff=None,
                 dropout=0.05, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.step = step_len
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        #self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm)
        self.lstm = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm)
        self.lstm1 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm)
        self.lstm2 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.decomp1 = series_decomp(step_len)
        self.decomp2 = series_decomp(step_len)
        self.decomp3 = series_decomp(step_len)
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        y1 = x.permute(1, 0, 2).to(torch.float32)
        y1, hidden = self.lstm(y1)
        y1 = y1.permute(1, 0, 2)
        y1 = self.dropout(torch.softmax(y1, dim=-1)*x) + x 
        x = self.dropout(self.self_attention(
            x, x, x, x_mask
        )[0]) + y1
        x = self.norm1(x)

        new_y = self.dropout(self.activation(self.conv3(x.transpose(-1,1))))
        new_y = self.dropout(self.conv4(new_y).transpose(-1,1))
        
        x, trend1 = self.decomp1(x)
        y1, _ = self.lstm1(x.permute(1, 0, 2))
        y1 = y1.permute(1, 0, 2)
        new_x = x + self.dropout(x*torch.softmax(y1, dim=-1)) + self.dropout(self.cross_attention(
            x, cross, cross, cross_mask
        )[0])
        x, trend2 = self.decomp2(new_x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        x, trend3 = self.decomp3(x + y)

        x = (x + new_y)/2
        residual_trend  = trend1 + trend2 + trend3
        residual_trend, _ = self.lstm2(residual_trend.permute(1, 0, 2))
        residual_trend = residual_trend.permute(1, 0, 2)
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return self.norm3(x), residual_trend, hidden 

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend, hidden = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        return x, trend, hidden