import torch
import torch.nn
import random
import math

class SMILESTransformer(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(SMILESTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.ninp = ninp

        self.pos_encoder = torch.nn.Embedding(100, ninp)
        self.encoder = torch.nn.Embedding(ntoken, ninp)

        self.layer_norm = torch.nn.LayerNorm([ninp])
        self.output_layer_norm = torch.nn.LayerNorm([ntoken])
        self.input_layer_norm = torch.nn.LayerNorm([ninp])

        encoder_layers = TransformerEncoderLayer(d_model=ninp,
                                                 nhead=nhead,
                                                 dim_feedforward=nhid,
                                                 dropout=dropout,
                                                 activation='gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      nlayers,
                                                      norm=self.layer_norm)

        self.dropout = torch.nn.Dropout(dropout)

        self.decoder = torch.nn.Linear(ninp, ntoken, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(ntoken))
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.normal_(mean=0.0, std=1.0)
        self.decoder.weight.data.normal_(mean=0.0, std=1.0)
        self.decoder_bias.data.zero_()

        self.input_layer_norm.weight.data.fill_(1.0)
        self.input_layer_norm.bias.data.zero_()
        self.output_layer_norm.weight.data.fill_(1.0)
        self.output_layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.layer_norm.bias.data.zero_()


    def forward(self, src, latent_out=False):
        pos = torch.arange(0,100).long().to(src.device)

        mol_token_emb = self.encoder(src)
        pos_emb = self.pos_encoder(pos)
        input_emb = pos_emb + mol_token_emb
        input_emb = self.input_layer_norm(input_emb) 
        input_emb = self.dropout(input_emb)
        input_emb = input_emb.transpose(0, 1) 

        attention_mask = torch.ones_like(src).to(src.device)
        attention_mask = attention_mask.masked_fill(src!=1., 0.)
        attention_mask = attention_mask.bool().to(src.device)

        output = self.transformer_encoder(input_emb)
        
        if latent_out:
            return output
        output = self.decoder(output) + self.decoder_bias 

        return output