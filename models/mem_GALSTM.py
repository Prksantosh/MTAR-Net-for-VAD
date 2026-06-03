# -*- coding: utf-8 -*-
"""
Memory-Augmented E3D-LSTM
UPDATED VERSION:
- Replaced Conv3D operations with Conv2D
- Temporal dimension processed independently
- Much lower memory usage
- Faster inference
- Better FPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################
# LayerNorm over last dimension
############################################################
class LayerNormLastDim(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        return self.norm(x)


############################################################
# Memory-Guided Temporal Attention
############################################################
class MemoryGuidedTemporalAttention(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=4,
        memory_slots=1000,
        dropout=0.1
    ):

        super().__init__()

        if channels % num_heads != 0:

            raise ValueError(
                f"channels ({channels}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.memory_slots = memory_slots

        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------

        self.norm = LayerNormLastDim(channels)

        # --------------------------------------------------
        # QKV projections
        # --------------------------------------------------

        self.q_proj = nn.Linear(
            channels,
            channels,
            bias=False
        )

        self.k_proj = nn.Linear(
            channels,
            channels,
            bias=False
        )

        self.v_proj = nn.Linear(
            channels,
            channels,
            bias=False
        )

        # --------------------------------------------------
        # Learnable Memory Bank
        # --------------------------------------------------

        self.memory = nn.Parameter(
            torch.randn(memory_slots, channels)
        )

        self.mem_k_proj = nn.Linear(
            channels,
            channels,
            bias=False
        )

        self.mem_v_proj = nn.Linear(
            channels,
            channels,
            bias=False
        )

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        self.out_proj = nn.Linear(
            channels,
            channels,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # Memory Fusion Gate
        # --------------------------------------------------

        self.gate_proj = nn.Linear(
            channels * 2,
            channels
        )

    def forward(self, x):

        """
        x: (B, C, T, H, W)
        """

        B, C, T, H, W = x.shape

        N = H * W

        # --------------------------------------------------
        # Convert to temporal tokens
        # --------------------------------------------------

        x_tokens = (
            x.permute(0, 3, 4, 2, 1)
             .contiguous()
             .view(B * N, T, C)
        )

        x_norm = self.norm(x_tokens)

        # --------------------------------------------------
        # QKV
        # --------------------------------------------------

        q = self.q_proj(x_norm)

        k = self.k_proj(x_norm)

        v = self.v_proj(x_norm)

        # --------------------------------------------------
        # Memory Tokens
        # --------------------------------------------------

        mem = self.memory.unsqueeze(0).expand(
            B * N,
            -1,
            -1
        )

        mem_k = self.mem_k_proj(mem)

        mem_v = self.mem_v_proj(mem)

        # --------------------------------------------------
        # Concatenate Memory
        # --------------------------------------------------

        k_all = torch.cat([k, mem_k], dim=1)

        v_all = torch.cat([v, mem_v], dim=1)

        # --------------------------------------------------
        # Multi-head reshape
        # --------------------------------------------------

        q = (
            q.view(
                B * N,
                T,
                self.num_heads,
                self.head_dim
            )
            .transpose(1, 2)
        )

        k_all = (
            k_all.view(
                B * N,
                T + self.memory_slots,
                self.num_heads,
                self.head_dim
            )
            .transpose(1, 2)
        )

        v_all = (
            v_all.view(
                B * N,
                T + self.memory_slots,
                self.num_heads,
                self.head_dim
            )
            .transpose(1, 2)
        )

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        attn = torch.matmul(
            q,
            k_all.transpose(-2, -1)
        ) * self.scale

        attn = torch.softmax(attn, dim=-1)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v_all)

        out = (
            out.transpose(1, 2)
               .contiguous()
               .view(B * N, T, C)
        )

        # --------------------------------------------------
        # Explicit Memory Read
        # --------------------------------------------------

        mem_attn = torch.matmul(
            self.q_proj(x_norm),
            self.memory.t()
        ) * (C ** -0.5)

        mem_attn = torch.softmax(mem_attn, dim=-1)

        mem_read = torch.matmul(
            mem_attn,
            self.memory
        )

        # --------------------------------------------------
        # Gated Fusion
        # --------------------------------------------------

        fused = torch.cat(
            [out, mem_read],
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate_proj(fused)
        )

        out = (
            gate * out +
            (1.0 - gate) * mem_read
        )

        # --------------------------------------------------
        # Final Projection
        # --------------------------------------------------

        out = self.out_proj(out)

        # --------------------------------------------------
        # Back to 5D tensor
        # --------------------------------------------------

        out = (
            out.view(B, H, W, T, C)
               .permute(0, 4, 3, 1, 2)
               .contiguous()
        )

        return out


############################################################
# Memory-Augmented E3D-LSTM Cell
# UPDATED: Conv3D -> Conv2D
############################################################
class MemE3DLSTMCell(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        mem_slots=150,
        kernel_size=3,
        num_heads=4
    ):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        # --------------------------------------------------
        # Conv2D Gates
        #
        # Input:
        # (B*T, C, H, W)
        # --------------------------------------------------

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding
        )

        # --------------------------------------------------
        # Learnable Memory Bank
        # --------------------------------------------------

        self.memory = nn.Parameter(
            torch.randn(mem_slots, hidden_channels)
        )

        # --------------------------------------------------
        # Query Projection
        # --------------------------------------------------

        self.query_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1
        )

        # --------------------------------------------------
        # Memory Projection
        # --------------------------------------------------

        self.mem_proj = nn.Linear(
            hidden_channels,
            hidden_channels
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        self.fusion = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=1
        )

        # --------------------------------------------------
        # Temporal Attention
        # --------------------------------------------------

        self.temporal_attention = (
            MemoryGuidedTemporalAttention(
                channels=hidden_channels,
                num_heads=num_heads,
                memory_slots=mem_slots
            )
        )

        # --------------------------------------------------
        # Memory Gate
        # --------------------------------------------------

        self.memory_gate = nn.Sequential(

            nn.Conv2d(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x, h_prev, c_prev):

        """
        x:      (B, C, T, H, W)
        h_prev: (B, C, T, H, W)
        """

        B, C, T, H, W = x.shape

        # --------------------------------------------------
        # Convert to 2D format
        # --------------------------------------------------

        x2d = (
            x.permute(0, 2, 1, 3, 4)
             .contiguous()
             .view(B * T, C, H, W)
        )

        h2d = (
            h_prev.permute(0, 2, 1, 3, 4)
                  .contiguous()
                  .view(B * T,
                        self.hidden_channels,
                        H,
                        W)
        )

        c2d = (
            c_prev.permute(0, 2, 1, 3, 4)
                  .contiguous()
                  .view(B * T,
                        self.hidden_channels,
                        H,
                        W)
        )

        # ==================================================
        # ConvLSTM Update
        # ==================================================

        combined = torch.cat(
            [x2d, h2d],
            dim=1
        )

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)

        f = torch.sigmoid(f)

        o = torch.sigmoid(o)

        g = torch.tanh(g)

        # --------------------------------------------------
        # LSTM update
        # --------------------------------------------------

        c = f * c2d + i * g

        h = o * torch.tanh(c)

        # ==================================================
        # Memory Attention
        # ==================================================

        query = self.query_conv(h)

        query = query.mean(dim=[2, 3])

        mem = self.mem_proj(self.memory)

        attn = torch.softmax(

            torch.matmul(
                query,
                mem.t()
            ),

            dim=-1
        )

        mem_read = torch.matmul(
            attn,
            mem
        )

        mem_read = (
            mem_read.view(
                B * T,
                self.hidden_channels,
                1,
                1
            )
            .expand(-1, -1, H, W)
        )

        # ==================================================
        # Fusion
        # ==================================================

        fused = torch.cat(
            [h, mem_read],
            dim=1
        )

        h_mem = self.fusion(fused)

        # --------------------------------------------------
        # Restore 5D for temporal attention
        # --------------------------------------------------

        h_mem_5d = (
            h_mem.view(
                B,
                T,
                self.hidden_channels,
                H,
                W
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        # ==================================================
        # Temporal Attention
        # ==================================================

        h_attn = self.temporal_attention(h_mem_5d)

        # --------------------------------------------------
        # Back to 2D
        # --------------------------------------------------

        h_attn = (
            h_attn.permute(0, 2, 1, 3, 4)
                  .contiguous()
                  .view(B * T,
                        self.hidden_channels,
                        H,
                        W)
        )

        # ==================================================
        # Gated Fusion
        # ==================================================

        gate_input = torch.cat(
            [h_mem, h_attn],
            dim=1
        )

        gate = self.memory_gate(gate_input)

        h_final = (
            gate * h_mem +
            (1.0 - gate) * h_attn
        )

        # --------------------------------------------------
        # Restore 5D
        # --------------------------------------------------

        h_final = (
            h_final.view(
                B,
                T,
                self.hidden_channels,
                H,
                W
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        c = (
            c.view(
                B,
                T,
                self.hidden_channels,
                H,
                W
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        return h_final, c


############################################################
# Memory-Augmented E3D-LSTM Layer
############################################################
class MemE3DLSTM(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        mem_slots=150,
        num_heads=4
    ):

        super().__init__()

        self.cell1 = MemE3DLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            mem_slots=mem_slots,
            num_heads=num_heads
        )

        self.cell2 = MemE3DLSTMCell(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            mem_slots=mem_slots,
            num_heads=num_heads
        )

    def forward(self, x):

        """
        x: (B, C, T, H, W)
        """

        B, C, T, H, W = x.shape

        device = x.device

        # --------------------------------------------------
        # Hidden States
        # --------------------------------------------------

        h1 = torch.zeros(
            B,
            self.cell1.hidden_channels,
            T,
            H,
            W,
            device=device
        )

        c1 = torch.zeros_like(h1)

        h2 = torch.zeros(
            B,
            self.cell2.hidden_channels,
            T,
            H,
            W,
            device=device
        )

        c2 = torch.zeros_like(h2)

        # --------------------------------------------------
        # Layer 1
        # --------------------------------------------------

        h1, c1 = self.cell1(
            x,
            h1,
            c1
        )

        # --------------------------------------------------
        # Layer 2
        # --------------------------------------------------

        h2, c2 = self.cell2(
            h1,
            h2,
            c2
        )

        return h2