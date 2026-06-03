import torch
import torch.nn as nn

###############################################
# ConvLSTM Cell
###############################################
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #################################
        # Gate Convolution
        #################################

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, h, c):

        # x : (B,C,H,W)
        # h : (B,hidden_dim,H,W)

        #################################
        # Fix channel mismatch
        #################################

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input channels = {self.input_dim}, "
                f"but got {x.shape[1]}"
            )

        if h.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden channels = {self.hidden_dim}, "
                f"but got {h.shape[1]}"
            )

        #################################
        # LSTM Gates
        #################################

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        g = torch.tanh(g)

        #################################
        # Cell Update
        #################################

        c = f * c + i * g

        h = o * torch.tanh(c)

        return h, c


###############################################
# ConvLSTM Layer
###############################################
class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.cell = ConvLSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x):

        # x : (B,T,C,H,W)

        B, T, C, H, W = x.shape

        #################################
        # Ensure input matches model config
        #################################

        if C != self.input_dim:
            raise ValueError(
                f"ConvLSTM expected {self.input_dim} input channels "
                f"but received {C}"
            )

        #################################
        # Initial hidden states
        #################################

        h = torch.zeros(
            B,
            self.hidden_dim,
            H,
            W,
            device=x.device
        )

        c = torch.zeros_like(h)

        outputs = []

        #################################
        # Temporal recurrence
        #################################

        for t in range(T):

            xt = x[:, t]

            h, c = self.cell(xt, h, c)

            outputs.append(h)

        #################################
        # Stack outputs
        #################################

        outputs = torch.stack(outputs, dim=1)

        return outputs, h