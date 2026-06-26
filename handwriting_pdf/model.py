"""PyTorch reimplementation of the handwriting synthesis RNN described in

  Alex Graves, "Generating Sequences With Recurrent Neural Networks"
  (https://arxiv.org/abs/1308.0850), section 5 (handwriting synthesis network).

Loads weights ported from the pretrained TensorFlow checkpoint published in
sjvasquez/handwriting-synthesis (https://github.com/sjvasquez/handwriting-synthesis),
which trained this exact architecture on IAM-OnDB.

Architecture: a 3-layer LSTM stack with a Gaussian-window soft-attention
mechanism reading a one-hot character sequence, sitting between layer 1 and
layer 2, and a mixture-density-network (MDN) output head reading the layer-3
hidden state to produce a distribution over the next pen offset (dx, dy) and
pen-lift probability.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import drawing

LSTM_SIZE = 400
NUM_ATTN_COMPONENTS = 10
NUM_OUTPUT_COMPONENTS = 20
ALPHABET_SIZE = len(drawing.alphabet)  # 73


class TFStyleLSTMCell(nn.Module):
    """An LSTM cell with the exact gate layout/order of TensorFlow's
    contrib.rnn.LSTMCell, so pretrained kernel/bias tensors load unchanged.

    gate order in the fused kernel is (i, j, f, o); forget gate gets a
    +1.0 bias before the sigmoid, matching TF's default forget_bias.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel = nn.Parameter(torch.empty(input_size + hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))

    def forward(self, x, h, c):
        gates = torch.cat([x, h], dim=1) @ self.kernel + self.bias
        i, j, f, o = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class HandwritingRNN(nn.Module):
    def __init__(
        self,
        lstm_size=LSTM_SIZE,
        num_attn_components=NUM_ATTN_COMPONENTS,
        num_output_components=NUM_OUTPUT_COMPONENTS,
        alphabet_size=ALPHABET_SIZE,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.num_attn_components = num_attn_components
        self.num_output_components = num_output_components
        self.alphabet_size = alphabet_size
        self.output_units = 6 * num_output_components + 1

        self.lstm1 = TFStyleLSTMCell(alphabet_size + 3, lstm_size)
        self.lstm2 = TFStyleLSTMCell(3 + lstm_size + alphabet_size, lstm_size)
        self.lstm3 = TFStyleLSTMCell(3 + lstm_size + alphabet_size, lstm_size)

        self.attn_weights = nn.Parameter(torch.empty(alphabet_size + 3 + lstm_size, 3 * num_attn_components))
        self.attn_bias = nn.Parameter(torch.empty(3 * num_attn_components))

        self.gmm_weights = nn.Parameter(torch.empty(lstm_size, self.output_units))
        self.gmm_bias = nn.Parameter(torch.empty(self.output_units))

    @classmethod
    def from_pretrained(cls, npz_path):
        weights = np.load(npz_path)
        model = cls()
        prefix = "rnn/LSTMAttentionCell/"
        with torch.no_grad():
            model.lstm1.kernel.copy_(torch.from_numpy(weights[prefix + "lstm_cell/kernel"]))
            model.lstm1.bias.copy_(torch.from_numpy(weights[prefix + "lstm_cell/bias"]))
            model.lstm2.kernel.copy_(torch.from_numpy(weights[prefix + "lstm_cell_1/kernel"]))
            model.lstm2.bias.copy_(torch.from_numpy(weights[prefix + "lstm_cell_1/bias"]))
            model.lstm3.kernel.copy_(torch.from_numpy(weights[prefix + "lstm_cell_2/kernel"]))
            model.lstm3.bias.copy_(torch.from_numpy(weights[prefix + "lstm_cell_2/bias"]))
            model.attn_weights.copy_(torch.from_numpy(weights[prefix + "attention/weights"]))
            model.attn_bias.copy_(torch.from_numpy(weights[prefix + "attention/biases"]))
            model.gmm_weights.copy_(torch.from_numpy(weights["rnn/gmm/weights"]))
            model.gmm_bias.copy_(torch.from_numpy(weights["rnn/gmm/biases"]))
        model.eval()
        return model

    def zero_state(self, batch_size, char_len, device):
        z = lambda *shape: torch.zeros(*shape, device=device)
        return dict(
            h1=z(batch_size, self.lstm_size), c1=z(batch_size, self.lstm_size),
            h2=z(batch_size, self.lstm_size), c2=z(batch_size, self.lstm_size),
            h3=z(batch_size, self.lstm_size), c3=z(batch_size, self.lstm_size),
            kappa=z(batch_size, self.num_attn_components),
            w=z(batch_size, self.alphabet_size),
            phi=z(batch_size, char_len),
        )

    def step(self, x, state, chars_onehot, char_lengths):
        """One timestep. x: [B, 3] pen input. chars_onehot: [B, char_len, alphabet_size]."""
        batch_size, char_len, _ = chars_onehot.shape

        s1_in = torch.cat([state["w"], x], dim=1)
        h1, c1 = self.lstm1(s1_in, state["h1"], state["c1"])

        attn_in = torch.cat([state["w"], x, h1], dim=1)
        attn_params = F.softplus(attn_in @ self.attn_weights + self.attn_bias)
        alpha, beta, kappa = attn_params.chunk(3, dim=1)
        kappa = state["kappa"] + kappa / 25.0
        beta = beta.clamp(min=0.01)

        u = torch.arange(char_len, device=x.device, dtype=torch.float32).view(1, 1, char_len)
        kappa_e = kappa.unsqueeze(2)
        alpha_e = alpha.unsqueeze(2)
        beta_e = beta.unsqueeze(2)
        phi_flat = (alpha_e * torch.exp(-torch.square(kappa_e - u) / beta_e)).sum(dim=1)  # [B, char_len]

        seq_mask = (torch.arange(char_len, device=x.device).unsqueeze(0) < char_lengths.unsqueeze(1)).float()
        w = torch.bmm((phi_flat * seq_mask).unsqueeze(1), chars_onehot).squeeze(1)  # [B, alphabet]

        s2_in = torch.cat([x, h1, w], dim=1)
        h2, c2 = self.lstm2(x=s2_in, h=state["h2"], c=state["c2"])

        s3_in = torch.cat([x, h2, w], dim=1)
        h3, c3 = self.lstm3(x=s3_in, h=state["h3"], c=state["c3"])

        new_state = dict(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3, kappa=kappa, w=w, phi=phi_flat)
        return new_state

    def output_params(self, state, bias):
        """Return MDN params (pis, mus, sigma1, sigma2, rho, es) from the current state."""
        K = self.num_output_components
        gmm = state["h3"] @ self.gmm_weights + self.gmm_bias
        pis, sigmas, rhos, mus, es = torch.split(gmm, [K, 2 * K, K, 2 * K, 1], dim=-1)

        pis = pis * (1 + bias.unsqueeze(1))
        sigmas = sigmas - bias.unsqueeze(1)

        pis = F.softmax(pis, dim=-1)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        sigmas = torch.exp(sigmas).clamp(min=1e-4)
        rhos = torch.tanh(rhos).clamp(min=-1 + 1e-8, max=1 - 1e-8)
        es = torch.sigmoid(es).clamp(min=1e-8, max=1 - 1e-8)
        es = torch.where(es < 0.01, torch.zeros_like(es), es)

        mu1, mu2 = mus.chunk(2, dim=1)
        sigma1, sigma2 = sigmas.chunk(2, dim=1)
        return pis, mu1, mu2, sigma1, sigma2, rhos, es
