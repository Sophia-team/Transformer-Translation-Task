import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import math
import random
import math
import time


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# class MyTransformer(nn.Module):
#     def __init__(
#       self,
#       input_dim,
#       output_dim,
#       emb_dim,
#       dim_feedforward,
#       num_encoder_layers, 
#       num_decoder_layers,  
#       dropout,  
#       src_pad_idx=None,
#       tgt_pad_idx=None,
#       src_mask=None,
#       tgt_mask=None,
#       ):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.emb_dim = emb_dim
#         self.dim_feedforward = dim_feedforward
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         self.src_mask = src_mask
#         self.tgt_mask = tgt_mask
#         self.src_pad_idx = src_pad_idx
#         self.tgt_pad_idx = tgt_pad_idx
#         self.src_key_padding_mask = None
#         self.tgt_key_padding_mask = None

#         self.encoder_embedding = nn.Embedding(
#             num_embeddings=input_dim,
#             embedding_dim=emb_dim
#         )

#         self.decoder_embedding = nn.Embedding(
#             num_embeddings=output_dim,
#             embedding_dim=emb_dim
#         )

#         self.pos_encoder = PositionalEncoding(d_model=emb_dim, dropout=dropout)

#         self.transformer = nn.Transformer(
#           d_model=emb_dim,
#           nhead=8, 
#           num_encoder_layers=num_encoder_layers, 
#           num_decoder_layers=num_decoder_layers,
#           dim_feedforward = dim_feedforward,
#           dropout=dropout)

#         self.out = nn.Linear(
#             in_features=emb_dim,
#             out_features=output_dim
#         )
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, src, trg):
#       if self.tgt_mask is None or self.tgt_mask.size(0) != len(trg)::
#           device = trg.device
#           mask = self._generate_square_subsequent_mask(len(trg)).to(device)
#           self.tgt_mask = mask
      
#       if (self.src_pad_idx is not None):
#         self.src_key_padding_mask = self._generate_key_padding_mask(src, self.src_pad_idx).to(device)

#       if (self.tgt_pad_idx is not None):
#         self.tgt_key_padding_mask = self._generate_key_padding_mask(trg, self.tgt_pad_idx).to(device)

#       src = self.dropout(self.encoder_embedding(src) * math.sqrt(self.emb_dim))
#       src = self.pos_encoder(src)
#       trg = self.dropout(self.decoder_embedding(trg) * math.sqrt(self.emb_dim))
#       trg = self.pos_encoder(trg)

#       output = self.transformer(
#        src, 
#        trg,
#        src_mask=self.src_mask,
#        tgt_mask=self.tgt_mask,
#        src_key_padding_mask=self.src_key_padding_mask,
#        tgt_key_padding_mask=self.tgt_key_padding_mask,
#        )
#       output = self.dropout(self.out(output))
#       return output

def _generate_square_subsequent_mask(sz):
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

def _generate_key_padding_mask(sz, pad_idx):
  tmp_sz = sz.cpu().clone()
  tmp_sz.apply_(lambda x: False if x != pad_idx else True)
  return torch.transpose(tmp_sz, 0, 1).bool()


class Encoder(nn.Module):
  def __init__(
    self,
    input_dim,
    emb_dim,
    nhead,
    num_encoder_layers,
    dim_feedforward,
    dropout,
    src_pad_idx,
    ):
    super().__init__()
    self.input_dim = input_dim
    self.emb_dim = emb_dim
    self.nhead = nhead
    self.num_encoder_layers = num_encoder_layers
    self.dim_feedforward = dim_feedforward
    self.src_pad_idx = src_pad_idx
    self.src_key_padding_mask = None

    self.encoder_embedding = nn.Embedding(
                num_embeddings=input_dim,
                embedding_dim=emb_dim
            )
    self.pos_encoder = PositionalEncoding(d_model=emb_dim, dropout=dropout)
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=emb_dim, 
      nhead=nhead,
      dim_feedforward=dim_feedforward
      )
    self.encoder = nn.TransformerEncoder(
      encoder_layer=self.encoder_layer, 
      num_layers=num_encoder_layers
      )
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, src):
    if (self.src_pad_idx is not None):
        device = src.device
        self.src_key_padding_mask = _generate_key_padding_mask(src, self.src_pad_idx).to(device)
    
    src = self.dropout(self.encoder_embedding(src) * math.sqrt(self.emb_dim))
    src = self.pos_encoder(src)

    output = self.encoder(
                src,
                src_key_padding_mask=self.src_key_padding_mask
                )
    return output


class Decoder(nn.Module):
  def __init__(
    self,
    output_dim,
    emb_dim,
    nhead,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    tgt_pad_idx
    ):
    super().__init__()

    self.output_dim = output_dim
    self.emb_dim = emb_dim
    self.nhead = nhead
    self.num_decoder_layers = num_decoder_layers
    self.dim_feedforward = dim_feedforward
    self.tgt_pad_idx = tgt_pad_idx
    self.tgt_key_padding_mask = None
    self.tgt_mask = None
    self.decoder_embedding = nn.Embedding(
                num_embeddings=output_dim,
                embedding_dim=emb_dim
            )
    self.pos_encoder = PositionalEncoding(d_model=emb_dim, dropout=dropout)
    self.decoder_layer = nn.TransformerDecoderLayer(
      d_model=emb_dim, 
      nhead=nhead,
      dim_feedforward=dim_feedforward
      )
    self.decoder = nn.TransformerDecoder(
      decoder_layer=self.decoder_layer, 
      num_layers=num_decoder_layers
      )
    self.dropout = nn.Dropout(p=dropout)
    self.out = nn.Linear(emb_dim, output_dim)

  def forward(self, tgt, memory, memory_key_padding_mask=None):

    # if (self.tgt_pad_idx is not None):
    #     device = tgt.device
    #     self.tgt_key_padding_mask = _generate_key_padding_mask(tgt, self.tgt_pad_idx).to(device)

    if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
          device = tgt.device
          mask = _generate_square_subsequent_mask(len(tgt)).to(device)
          self.tgt_mask = mask

    tgt = self.dropout(self.decoder_embedding(tgt) * math.sqrt(self.emb_dim))

    tgt = self.pos_encoder(tgt)

    output = self.decoder(
                tgt,
                memory,
                tgt_mask=self.tgt_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
                )
    output = self.out(output)
    return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # assert encoder.hid_dim == decoder.hid_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert encoder.n_layers == decoder.n_layers, \
        #     "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #tensor to store decoder outputs
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        #first input to the decoder is the <sos> tokens
        encoder_output = self.encoder(src)
        input = trg[0,:].unsqueeze(0)
        memory_key_padding_mask=self.encoder.src_key_padding_mask
       
        for t in range(1, max_len):
            output = self.decoder(input, encoder_output,
             memory_key_padding_mask=memory_key_padding_mask)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(2)[1]
            new_input = (trg[t] if teacher_force else top1[-1])
            input = torch.cat((input, new_input.unsqueeze(0)), dim=0)
        outputs[1:] = output
        return outputs

