import torch
import torch.nn as nn

class BERT4Rec(nn.Module):
    def __init__(self, item_num, max_len, hidden_units, num_heads, num_layers, dropout):
        super(BERT4Rec, self).__init__()
        self.item_num = item_num
        self.max_len = max_len
        self.hidden_units = hidden_units
        
        # Item embedding layer
        self.item_emb = nn.Embedding(item_num + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, 
            nhead=num_heads, 
            dim_feedforward=hidden_units * 4, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True  # Set batch_first to True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.out = nn.Linear(hidden_units, item_num + 1)
        
    def forward(self, x):
        x = self.item_emb(x)
        
        # Add positional embeddings
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x += self.pos_emb(positions)
        
        # Create mask for padding
        mask = self.get_attention_mask(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return self.out(x)
    
    def get_attention_mask(self, x):
        # Create a mask for padding tokens (assuming 0 is the padding token)
        mask = (x == 0).all(dim=-1)
        return mask

