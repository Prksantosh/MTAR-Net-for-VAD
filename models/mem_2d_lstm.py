import torch
import torch.nn as nn

###############################################
# Memory-Augmented E3D-LSTM Cell (2D Version)
###############################################
class MemE3DLSTMCell(nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 mem_slots=16, kernel_size=3):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels
        self.mem_slots = mem_slots

        #################################
        # LSTM Gates (2D)
        #################################

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=padding
        )

        #################################
        # Memory Bank
        #################################

        self.memory = nn.Parameter(
            torch.randn(mem_slots, hidden_channels)
        )

        #################################
        # Query Projection
        #################################

        self.query_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            1
        )

        #################################
        # Memory Projection
        #################################

        self.mem_proj = nn.Linear(
            hidden_channels,
            hidden_channels
        )

        #################################
        # Fusion
        #################################

        self.fusion = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels,
            1
        )

    def forward(self, x, h_prev, c_prev):

        # x : (B,C,H,W)

        combined = torch.cat([x, h_prev], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        #################################
        # LSTM Update
        #################################

        c = f * c_prev + i * g

        h = o * torch.tanh(c)

        #################################
        # Memory Attention
        #################################

        B, C, H, W = h.shape

        query = self.query_conv(h)

        query = query.mean(dim=[2, 3])  # (B,C)

        mem = self.mem_proj(self.memory)

        attn = torch.softmax(
            torch.matmul(query, mem.t()),
            dim=-1
        )

        mem_read = torch.matmul(attn, mem)

        mem_read = mem_read.view(
            B, C, 1, 1
        ).expand(-1, -1, H, W)

        #################################
        # Fusion
        #################################

        fused = torch.cat([h, mem_read], dim=1)

        h = self.fusion(fused)

        return h, c


###############################################
# Memory-Augmented E3D-LSTM Layer
###############################################
class MemE3DLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels):

        super().__init__()

        self.cell1 = MemE3DLSTMCell(
            in_channels,
            hidden_channels
        )

        self.cell2 = MemE3DLSTMCell(
            hidden_channels,
            hidden_channels
        )

    def forward(self, x):

        # x : (B,C,T,H,W)

        B, C, T, H, W = x.shape

        device = x.device

        #################################
        # Initial States
        #################################

        h1 = torch.zeros(
            B,
            self.cell1.hidden_channels,
            H,
            W,
            device=device
        )

        c1 = torch.zeros_like(h1)

        h2 = torch.zeros(
            B,
            self.cell2.hidden_channels,
            H,
            W,
            device=device
        )

        c2 = torch.zeros_like(h2)

        outputs = []

        #################################
        # Process sequence frame-by-frame
        #################################

        for t in range(T):

            xt = x[:, :, t]   # (B,C,H,W)

            h1, c1 = self.cell1(
                xt,
                h1,
                c1
            )

            h2, c2 = self.cell2(
                h1,
                h2,
                c2
            )

            outputs.append(h2)

        #################################
        # Stack temporal outputs
        #################################

        outputs = torch.stack(outputs, dim=2)

        # (B,C,T,H,W)

        return outputs