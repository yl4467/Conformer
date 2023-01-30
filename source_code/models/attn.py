import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = False
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return (V.contiguous(), None)

class fft_attention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None):
        super(fft_attention, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        #corr.view(B, L, -1)
        #print(corr.shape, values.shape)
        out =  torch.einsum("bhls,bshd->bslh", corr, values)
        #print(out.shape)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

class LongformerSelfAttention(nn.Module):
    def __init__(self, num_attention_heads=8, d_model=512, attention_window=2, attention_dilation=1):
        super(LongformerSelfAttention, self).__init__()
        self.num_heads = num_attention_heads
        self.head_dim = int(d_model / num_attention_heads)
        self.embed_dim = d_model

        self.query = nn.Linear(d_model, self.embed_dim)
        self.key = nn.Linear(d_model, self.embed_dim)
        self.value = nn.Linear(d_model, self.embed_dim)

        self.query_global = nn.Linear(d_model, self.embed_dim)
        self.key_global = nn.Linear(d_model, self.embed_dim)
        self.value_global = nn.Linear(d_model, self.embed_dim)

        self.dropout = nn.Dropout(0.05)
        self.softmax = nn.Softmax()
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.attention_mode = 'sliding_chunks'

    def _skew(self, x, direction, padding_value):
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    def _skew2(self,x, padding_value):
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    def _chunk(self, x, w):
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    def sliding_chunks_matmul_qk(self, q, k, w, padding_value):
        bsz, seqlen, num_heads, head_dim = q.shape
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
        k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk(q, w)
        chunk_k = self._chunk(k, w)
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', chunk_q, chunk_k)  # multiply
        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)
        #mask_invalid_locations(diagonal_attn, w, 1, False)
        return diagonal_attn.cuda()


    def sliding_chunks_matmul_pv(self, prob, v, w):
        bsz, seqlen, num_heads, head_dim = v.size()
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
        v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
        padded_v = F.pad(v, (0, 0, w, w), value=2)
        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

        skewed_prob = self._skew2(chunk_prob, padding_value=1)

        context = torch.einsum('bcwd,bcdh->bcwh', skewed_prob, chunk_v)
        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


    def forward(self, query, key, value, attention_mask):
        #attention_mask = torch.tensor(attention_mask.squeeze(dim=2).squeeze(dim=1), dtype=float).cpu()
        #extra_attention_mask = attention_mask > 0
        #num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
        #max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
        #print(max_num_extra_indices_per_batch)
        #max_num_extra_indices_per_batch = int(attention_mask.shape[1]/5)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        seq_len, bsz, embed_dim = query.size()
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        attn_weights = self.sliding_chunks_matmul_qk(q, k, self.attention_window, 2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        #attn_weights = self.dropout(torch.softmax(attn_weights, dim=-1))
        #print(attn_weights)
        #attention_mask = attention_mask.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        
        #float_mask = attention_mask.new_ones(size=attention_mask.size())
        #mask = self.sliding_chunks_matmul_qk(float_mask, attention_mask, self.attention_window, 2)
        #mask = self.dropout(torch.softmax(mask, dim=-1))
        #attn_weights += mask
        #attn_weights = self.dropout(torch.softmax(attn_weights, dim=-1))
        #print(attn_weights)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = self.sliding_chunks_matmul_pv(attn_weights, v, self.attention_window)
        #attn = self.dropout(torch.softmax(attn, dim=-1))
        attn = attn.transpose(0, 1).reshape(bsz, seq_len, embed_dim).contiguous()
        #print(attn)
        """
        selected_query = query.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)

        q = self.query_global(selected_query)
        k = self.key_global(key)
        v = self.value_global(value)

        q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.dropout(torch.softmax(attn_weights, dim=-1))
        attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
        selected_attn = torch.bmm(attn_weights, v)
        selected_attn_4d = selected_attn.view(bsz, max_num_extra_indices_per_batch, -1)
        selected_attn_tmp = torch.cat((selected_attn_4d,torch.zeros(bsz, seq_len-max_num_extra_indices_per_batch, embed_dim).cuda()),dim=1)
        attn = attn + selected_attn_tmp
        """
        context_layer = attn
        outputs = context_layer 
        return outputs

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)
