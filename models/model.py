import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer, series_decomp
from models.attn import FullAttention, ProbAttention, AttentionLayer, LongformerSelfAttention, fft_attention
from models.embed import DataEmbedding

class normal_flow_layer(nn.Module):
    def __init__(self,d_model, c_out, out_len):
        super(normal_flow_layer,self).__init__()
        self.pred_len = out_len
        self.conv = nn.Sequential(nn.Conv1d(c_out,c_out,2), nn.ReLU())
        self.mu = nn.Linear(d_model+out_len, out_len)
        self.sigma = nn.Linear(d_model+out_len, out_len)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(0.05)
        self.relu = nn.ReLU()
    
    def forward(self,input_data, sample):
        h = self.conv(input_data)
        h = torch.cat((h[:,:,0:1],h),2)
        mu = self.mu(h)
        sigma = self.sigma(h)
        sample = mu + torch.exp(.5*sigma) * sample*0.1 
        h = torch.cat((h[:,:, 0:h.shape[2]-self.pred_len], sample), 2)
        output = input_data + 0.1*h
        return output, sample

class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, step_len, d_model=512, n_heads=8, e_layers=3, d_layers=2, 
                normal_layers=1, enc_lstm=1, dec_lstm=1, weight=0.2,window=2, d_ff=512, dropout=0.0, attn='long', embed='fixed', freq='h', activation='gelu', 
                distil=True, mix=True, device=torch.device('cuda:0')):
        super(Model, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.c_out = c_out
        self.output_attention = output_attention
        self.label_len = label_len

        self.distribution_dec_mu = nn.Linear(d_model, out_len)
        self.distribution_dec_presigma = nn.Linear(d_model, out_len)
        self.distribution_enc_mu = nn.Linear(d_model, out_len)
        self.distribution_enc_presigma = nn.Linear(d_model, out_len)
        self.distribution_sigma = nn.Softplus()
        self.batch_norm = nn.BatchNorm1d(out_len)
        self.decomp = series_decomp(step_len)
        
        self.enc_fix = nn.Linear(enc_lstm, c_out)
        self.dec_fix = nn.Linear(dec_lstm, c_out)
        self.weight = weight

        self.enc_embedding = DataEmbedding(seq_len, enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(label_len + out_len, dec_in, d_model, embed, freq, dropout)
        
        if attn =='long':
            Attn = LongformerSelfAttention(n_heads, d_model, attention_window=window, attention_dilation=1)
        elif attn == 'full':
            Attn = AttentionLayer(FullAttention(True, attention_dropout=dropout),
                d_model, n_heads, mix=mix)
        else:
            Attn = fft_attention(d_model, n_heads)
        #normal_flow
        self.normal_flow = nn.ModuleList([normal_flow_layer(d_model, c_out, out_len) for l in range(normal_layers)])
        #Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(attn, Attn, enc_lstm, step_len, d_model, d_ff,
                    dropout=dropout,
                    activation=activation) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, attention_dropout=dropout),           
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), 
                                d_model, n_heads, mix=False),
                    dec_lstm, step_len, d_model, c_out, d_ff, dropout=dropout, activation=activation,)
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        #projection
        self.projection = nn.Linear(d_model, c_out, bias=True)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        trend_init = torch.zeros_like(x_enc).float()
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
    
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, enc_hidden = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        dec_out, trend, dec_hidden = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        dec_out = self.projection(dec_out)
        dec_out = dec_out + trend

        enc_hidden_permute = enc_hidden.permute(1, 2, 0)
        dec_hidden_permute = dec_hidden.permute(1, 2, 0)
        enc_hidden_permute = self.enc_fix(enc_hidden_permute).permute(0, 2, 1)
        dec_hidden_permute = self.dec_fix(dec_hidden_permute).permute(0, 2, 1)
        
        enc_mu = self.distribution_enc_mu(enc_hidden_permute)
        enc_pre_sigma = self.distribution_enc_presigma(enc_hidden_permute)
        enc_sigma = self.distribution_sigma(enc_pre_sigma)
    
        dec_mu = self.distribution_dec_mu(dec_hidden_permute)
        dec_pre_sigma = self.distribution_dec_presigma(dec_hidden_permute)
        dec_sigma = self.distribution_sigma(dec_pre_sigma)
        
        eps = torch.normal(mean = 0, std = 1, size=enc_mu.shape).cuda()
        sample = enc_mu + torch.exp(.5*enc_sigma) * eps 
        sample = dec_mu + torch.exp(.5*dec_sigma) * sample*0.1

        h = torch.cat((dec_hidden_permute, sample), 2)
        for flow in self.normal_flow:
            h, sample = flow(h, sample)
        
        sample = sample.permute(0,2,1)
        sample = sample*self.weight + dec_out[:, -self.pred_len:, -self.c_out:]*(1-self.weight)
        return  sample
