import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalSequenceClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type == "rnn":
            self.seq = nn.RNN(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            out_dim = hidden_dim * 2
        elif self.model_type == "lstm":
            self.seq = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            out_dim = hidden_dim * 2
        elif self.model_type == "gru":
            self.seq = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            out_dim = hidden_dim * 2
        elif self.model_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.seq = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            out_dim = hidden_dim
        else:
            raise ValueError(f"Modèle séquentiel inconnu : {model_type}")

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "transformer":
            x = self.pos_enc(self.input_proj(x))
            out = self.seq(x).mean(dim=1) 
        else:
            out, _ = self.seq(x)
            out = out.mean(dim=1)         
        return self.head(out)