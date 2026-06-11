"""Lane-token Transformer Q-network (Scheme T, full version minus static geometry).

Drop-in replacement for the MLP Qnet in dqn_agent_txw.DQN (via its q_net_factory
hook): same flat-vector in, same (B, action_dim) Q-values out, so the whole DQN
training loop (vanilla / Double / PER), checkpointing and ε-greedy action selection
are untouched. The action space stays Discrete(num_green_phases) — the CLS readout
maps to K fixed Q-values; this is NOT a FRAP/phase-scorer head.

Input layout (must match PriorityLaneTokenObservationFunction.token_layout()):
    [ header (header_dim) | lane_0 token (token_dim) | ... | lane_{N-1} token ]
      legacy  : header = [phase_onehot(K), min_green]      token = φ
      perphase: header = [min_green_ok, phase_elapsed]     token = [is_green, φ]

Architecture:
    token --Linear--> d_model --+CLS--> TransformerEncoder(L layers, H heads, FF)
    Q = MLP( concat(CLS_out, header) ) -> action_dim
By design there is NO static lane-geometry block in the tokens (user decision);
lane identity is therefore exchangeable — what distinguishes lanes is their live φ
content (+ is_green in perphase mode), and the global phase context enters via the
header at the readout. dropout=0 keeps the net deterministic (this codebase never
toggles train/eval mode on the Q-nets).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LaneTokenTransformerQNet(nn.Module):
    def __init__(self, header_dim: int, n_tokens: int, token_dim: int, action_dim: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.header_dim = header_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        self.token_proj = nn.Linear(token_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + header_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        header = x[:, :self.header_dim]
        tokens = x[:, self.header_dim:].reshape(B, self.n_tokens, self.token_dim)
        h = self.token_proj(tokens)
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)   # prepend CLS
        h = self.encoder(h)
        cls_out = h[:, 0]
        return self.head(torch.cat([cls_out, header], dim=-1))

    @torch.no_grad()
    def cls_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Diagnostic: last-layer CLS→token attention, shape (B, n_tokens). Eval-only.

        Runs the stock encoder layers up to the last one, then recomputes that layer's
        self-attention with need_weights=True on its exact input (norm_first=False ⇒
        attention is applied to the raw layer input, so the weights match the ones used
        inside). Heads are averaged. Row 0 = CLS query; column 0 (CLS→CLS) is dropped.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        tokens = x[:, self.header_dim:].reshape(B, self.n_tokens, self.token_dim)
        h = self.token_proj(tokens)
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        layers = self.encoder.layers
        for layer in layers[:-1]:
            h = layer(h)
        _, w = layers[-1].self_attn(h, h, h, need_weights=True)   # (B, L, L), heads averaged
        return w[:, 0, 1:]
