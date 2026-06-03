import torch
import torch.nn as nn


class MemE3DLSTMCell(nn.Module):

    def __init__(self, in_channels, hidden_channels, mem_slots=16, kernel_size=3):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels
        self.mem_slots = mem_slots


        self.conv = nn.Conv3d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=padding
        )


        self.memory = nn.Parameter(
            torch.randn(mem_slots, hidden_channels)
        )


        self.query_conv = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            1
        )


        self.mem_proj = nn.Linear(hidden_channels, hidden_channels)


        self.fusion = nn.Conv3d(
            hidden_channels * 2,
            hidden_channels,
            1
        )

    def forward(self, x, h_prev, c_prev):

        combined = torch.cat([x, h_prev], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)


        c = f * c_prev + i * g
        h = o * torch.tanh(c)



        B, C, T, H, W = h.shape

        query = self.query_conv(h)

        query = query.mean(dim=[2,3,4])   

        mem = self.mem_proj(self.memory)  

        attn = torch.softmax(
            torch.matmul(query, mem.t()),
            dim=-1
        )

        mem_read = torch.matmul(attn, mem)  

        mem_read = mem_read.view(B, C, 1, 1, 1).expand(-1,-1,T,H,W)

    

        fused = torch.cat([h, mem_read], dim=1)

        h = self.fusion(fused)

        return h, c

class MemE3DLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels):

        super().__init__()

        self.cell1 = MemE3DLSTMCell(in_channels, hidden_channels)
        self.cell2 = MemE3DLSTMCell(hidden_channels, hidden_channels)

    def forward(self, x):



        B, C, T, H, W = x.shape

        device = x.device

        h1 = torch.zeros(B, self.cell1.hidden_channels, T, H, W, device=device)
        c1 = torch.zeros_like(h1)

        h2 = torch.zeros(B, self.cell2.hidden_channels, T, H, W, device=device)
        c2 = torch.zeros_like(h2)

        h1, c1 = self.cell1(x, h1, c1)

        h2, c2 = self.cell2(h1, h2, c2)

        return h2
