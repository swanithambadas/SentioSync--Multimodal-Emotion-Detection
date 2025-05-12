import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#MODEL DEFINITIONS
    #chunked VIT model
class ChunkedMultiStageViT(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        face_chunks: int = 8,
        pose_dim: int = 34,
        pose_chunks: int = 2,
        hidden_dim: int = 256,
        num_classes: int = 7,
        n_heads: int = 4,
        face_layers: int = 2,
        pose_layers: int = 2,
        fusion_layers: int = 4,
    ):
        super().__init__()
        assert face_dim % face_chunks == 0, "face_dim must divide evenly by face_chunks"
        assert pose_dim % pose_chunks == 0, "pose_dim must divide evenly by pose_chunks"

        self.face_chunks = face_chunks
        self.pose_chunks = pose_chunks
        self.f_chunk_size = face_dim // face_chunks
        self.p_chunk_size = pose_dim // pose_chunks

        self.face_proj = nn.Linear(self.f_chunk_size, hidden_dim)
        self.pose_proj = nn.Linear(self.p_chunk_size, hidden_dim)

        face_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.face_enc = nn.TransformerEncoder(face_enc_layer, num_layers=face_layers)

        pose_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.pose_enc = nn.TransformerEncoder(pose_enc_layer, num_layers=pose_layers)

        total_tokens = 1 + face_chunks + pose_chunks
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.fusion_pos = nn.Parameter(torch.randn(1, total_tokens, hidden_dim))

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.fusion_enc = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, face, pose):
        B = face.size(0)

        f = face.view(B, self.face_chunks, self.f_chunk_size)        
        f = self.face_proj(f)                                        
        f = self.face_enc(f)                                         

        p = pose.view(B, self.pose_chunks, self.p_chunk_size)        
        p = self.pose_proj(p)                                        
        p = self.pose_enc(p)                                        

        cls = self.cls_token.expand(B, -1, -1)                       
        seq = torch.cat([cls, f, p], dim=1)                          
        seq = seq + self.fusion_pos                                 

        fused = self.fusion_enc(seq)                                 
        cls_out = fused[:, 0]                                        
        return self.mlp_head(cls_out)

#model 2
class ChunkedMultiStageViT_fine(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        face_chunks: int = 8,
        pose_dim: int = 34,
        pose_chunks: int = 2,
        hidden_dim: int = 256,
        num_classes: int = 7,
        n_heads: int = 4,
        face_layers: int = 2,
        pose_layers: int = 2,
        fusion_layers: int = 4,
        dropout: float = 0.2,           
    ):
        super().__init__()
        assert face_dim % face_chunks == 0, "face_dim must divide evenly by face_chunks"
        assert pose_dim % pose_chunks == 0, "pose_dim must divide evenly by pose_chunks"

        self.face_chunks = face_chunks
        self.pose_chunks = pose_chunks
        self.f_chunk_size = face_dim // face_chunks
        self.p_chunk_size = pose_dim // pose_chunks

        # projections + per-modality encoders
        self.face_proj = nn.Linear(self.f_chunk_size, hidden_dim)
        self.pose_proj = nn.Linear(self.p_chunk_size, hidden_dim)
        fe = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        pe = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.face_enc = nn.TransformerEncoder(fe, num_layers=face_layers)
        self.pose_enc = nn.TransformerEncoder(pe, num_layers=pose_layers)

        # fusion
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.fusion_pos = nn.Parameter(torch.randn(1, 1 + face_chunks + pose_chunks, hidden_dim))
        fusion_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.fusion_enc = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)

        # a bit more dropout before head
        self.dropout = nn.Dropout(dropout)

        # MLP head (already had dropout, but we’ve harmonized rates)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, face, pose):
        B = face.size(0)

        # chunk + encode face
        f = face.view(B, self.face_chunks, self.f_chunk_size)
        f = self.face_proj(f)
        f = self.dropout(f)
        f = self.face_enc(f)

        # chunk + encode pose
        p = pose.view(B, self.pose_chunks, self.p_chunk_size)
        p = self.pose_proj(p)
        p = self.dropout(p)
        p = self.pose_enc(p)

        # prep cls token + fuse
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, f, p], dim=1) + self.fusion_pos
        fused = self.fusion_enc(seq)

        # take cls, regularize, then head
        cls_out = fused[:, 0]
        cls_out = self.dropout(cls_out)
        return self.mlp_head(cls_out)


#model 3
class ChunkedCrossAttnViT(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        face_chunks: int = 8,
        pose_dim: int = 34,
        pose_chunks: int = 2,
        hidden_dim: int = 256,
        num_classes: int = 7,
        n_heads: int = 4,
        face_layers: int = 2,
        pose_layers: int = 2,
        fusion_layers: int = 4,
    ):
        super().__init__()
        assert face_dim % face_chunks == 0, "face_dim must be divisible by face_chunks"
        assert pose_dim % pose_chunks == 0, "pose_dim must be divisible by pose_chunks"
        self.face_chunks = face_chunks
        self.pose_chunks = pose_chunks
        self.f_chunk_size = face_dim // face_chunks
        self.p_chunk_size = pose_dim // pose_chunks

        self.face_proj = nn.Linear(self.f_chunk_size, hidden_dim)
        self.pose_proj = nn.Linear(self.p_chunk_size, hidden_dim)

        fe = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.face_enc = nn.TransformerEncoder(fe, num_layers=face_layers)
        pe = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.pose_enc = nn.TransformerEncoder(pe, num_layers=pose_layers)

        self.f2p_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.p2f_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        total_tokens = 1 + face_chunks + pose_chunks
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, total_tokens, hidden_dim))

        fu = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.fusion_enc = nn.TransformerEncoder(fu, num_layers=fusion_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, face: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        B = face.size(0)

        f = face.view(B, self.face_chunks, self.f_chunk_size)
        f = self.face_proj(f)
        f = self.face_enc(f)

        p = pose.view(B, self.pose_chunks, self.p_chunk_size)
        p = self.pose_proj(p)
        p = self.pose_enc(p)

        f2p, _ = self.f2p_attn(query=f, key=p, value=p)
        f = f + f2p
        p2f, _ = self.p2f_attn(query=p, key=f, value=f)
        p = p + p2f

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, f, p], dim=1) + self.pos_emb

        fused = self.fusion_enc(seq)
        cls_out = fused[:, 0]
        return self.mlp_head(cls_out)

#model 4
class ChunkedCrossAttnViT_tuned(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        face_chunks: int = 8,
        pose_dim: int = 34,
        pose_chunks: int = 2,
        hidden_dim: int = 256,
        num_classes: int = 7,
        n_heads: int = 4,
        face_layers: int = 2,
        pose_layers: int = 2,
        fusion_layers: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        assert face_dim % face_chunks == 0
        assert pose_dim % pose_chunks == 0

        self.face_chunks = face_chunks
        self.pose_chunks = pose_chunks
        self.f_chunk_size = face_dim // face_chunks
        self.p_chunk_size = pose_dim // pose_chunks

        self.face_proj = nn.Linear(self.f_chunk_size, hidden_dim)
        self.pose_proj = nn.Linear(self.p_chunk_size, hidden_dim)

        fe = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.face_enc = nn.TransformerEncoder(fe, num_layers=face_layers)
        pe = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.pose_enc = nn.TransformerEncoder(pe, num_layers=pose_layers)

        self.f2p_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.p2f_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        total_tokens = 1 + face_chunks + pose_chunks
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, total_tokens, hidden_dim))

        fu = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.fusion_enc = nn.TransformerEncoder(fu, num_layers=fusion_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # <-- Apply dropout here
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, face: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        B = face.size(0)

        f = face.view(B, self.face_chunks, self.f_chunk_size)
        f = self.face_proj(f)
        f = self.face_enc(f)

        p = pose.view(B, self.pose_chunks, self.p_chunk_size)
        p = self.pose_proj(p)
        p = self.pose_enc(p)

        f2p, _ = self.f2p_attn(query=f, key=p, value=p)
        f = f + f2p
        p2f, _ = self.p2f_attn(query=p, key=f, value=f)
        p = p + p2f

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, f, p], dim=1) + self.pos_emb

        fused = self.fusion_enc(seq)
        cls_out = fused[:, 0]
        return self.mlp_head(cls_out)