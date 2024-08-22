import torch
from torch import nn

from model import InputEmbeddings, PositionalEncoding, MultiHeadAttentionBlock, FeedForwardBlock, ResidualConnection, DecoderBlock, LayerNormalization, ProjectionLayer


class GPTBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class GPTDecoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class GPTModel(nn.Module):
    def __init__(self, decoder: GPTDecoder, embed: InputEmbeddings, pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.embed = embed
        self.pos = pos
        self.decoder = decoder
        self.projection_layer = projection_layer

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos(x)
        x = self.decoder(x, mask)
        return x
        
    def project(self, x):
        x = self.projection_layer(x)        
        return x 
    


def build_GPT(vocab_size: int, seq_len: int, 
                      d_model: int = 768, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 1024) -> GPTModel:
    # N should be > 12
    
    # Create embeddings
    embed = InputEmbeddings(d_model, vocab_size)
    
    # Create POS encodings 
    pos = PositionalEncoding(d_model, seq_len, dropout)

    # Create GPT Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = GPTBlock(decoder_self_attention, decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create GPT Decoder
    decoder = GPTDecoder(nn.ModuleList(decoder_blocks))
    
    # Create proj
    projection_layer = ProjectionLayer(d_model, vocab_size)
    
    # Create GPT model
    GPT = GPTModel(decoder, embed, pos, projection_layer)
    
    # Initialize parameters
    for p in GPT.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return GPT

