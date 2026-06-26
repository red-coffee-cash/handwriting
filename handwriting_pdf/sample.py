"""Autoregressive sampling from the handwriting RNN.

Generates pen-stroke offsets (dx, dy, end-of-stroke) one timestep at a time,
feeding each sampled output back in as the next input, until the attention
mechanism has swept past the end of the requested text (or a max length is
hit). Optionally "primes" the model on a short real stroke sample first
(see Graves 2013, section 5.3) to bias the generated handwriting toward a
particular style.
"""
import os

import numpy as np
import torch

import drawing
from model import HandwritingRNN

CHARS_PER_TIMESTEP = 40
MAX_TIMESTEPS = 1800

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "model_weights.npz")
_STYLE_STROKES_PATH = os.path.join(os.path.dirname(__file__), "weights", "style-9-strokes.npy")
_STYLE_CHARS_PATH = os.path.join(os.path.dirname(__file__), "weights", "style-9-chars.npy")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = HandwritingRNN.from_pretrained(_WEIGHTS_PATH)
    return _model


def _one_hot_chars(char_ids, alphabet_size):
    char_ids = torch.as_tensor(char_ids, dtype=torch.long)
    return torch.nn.functional.one_hot(char_ids, num_classes=alphabet_size).float()


def _sample_mdn(pis, mu1, mu2, sigma1, sigma2, rho, es):
    """Sample (dx, dy, eos) for a batch of timesteps given MDN parameters."""
    batch_size = pis.shape[0]
    component = torch.multinomial(pis, num_samples=1).squeeze(1)  # [B]
    idx = torch.arange(batch_size)

    m1 = mu1[idx, component]
    m2 = mu2[idx, component]
    s1 = sigma1[idx, component]
    s2 = sigma2[idx, component]
    r = rho[idx, component]

    z1 = torch.randn(batch_size)
    z2 = torch.randn(batch_size)
    x = m1 + s1 * z1
    y = m2 + s2 * (r * z1 + torch.sqrt((1 - r ** 2).clamp(min=1e-8)) * z2)

    eos = torch.bernoulli(es.squeeze(1))
    return torch.stack([x, y, eos], dim=1)


@torch.no_grad()
def sample_strokes(text, bias=0.75, style_prime=True, max_timesteps=None, seed=None):
    """Generate a stroke sequence (N, 3) of (dx, dy, eos) for one line of text.

    bias: in [0, ~1+], higher values bias the mixture toward higher-probability
        (neater, less varied) strokes. 0 = unbiased, ~0.7-1.0 = neat handwriting.
    style_prime: if True, primes the model on a bundled reference handwriting
        sample so output looks like a consistent, "trained" cursive style
        rather than the unconditional default.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = get_model()
    alphabet_size = model.alphabet_size
    device = torch.device("cpu")

    if style_prime:
        prime_strokes = np.load(_STYLE_STROKES_PATH).astype(np.float32)
        prime_chars_raw = np.load(_STYLE_CHARS_PATH)
        prime_chars = bytes(prime_chars_raw.tolist()).decode("utf-8")
        full_text = prime_chars + " " + text
        prime_len = len(prime_strokes)
    else:
        prime_strokes = None
        full_text = text
        prime_len = 0

    char_ids = drawing.encode_ascii(full_text)
    char_len = len(char_ids)
    chars_onehot = _one_hot_chars(char_ids, alphabet_size).unsqueeze(0).to(device)  # [1, char_len, A]
    char_lengths = torch.tensor([char_len], dtype=torch.long)
    bias_t = torch.tensor([bias], dtype=torch.float32)

    state = model.zero_state(batch_size=1, char_len=char_len, device=device)

    if prime_strokes is not None:
        for t in range(prime_len):
            x_t = torch.from_numpy(prime_strokes[t]).unsqueeze(0)
            state = model.step(x_t, state, chars_onehot, char_lengths)
        # Seed free-running generation with a sample drawn from the primed
        # state, rather than a fresh start token, so the transition from the
        # priming sample into the generated text is smooth.
        pis, mu1, mu2, sigma1, sigma2, rho, es = model.output_params(state, bias_t)
        x = _sample_mdn(pis, mu1, mu2, sigma1, sigma2, rho, es)
    else:
        x = torch.cat([torch.zeros(1, 2), torch.ones(1, 1)], dim=1)  # initial input: pen-up at origin

    max_t = max_timesteps or min(MAX_TIMESTEPS, CHARS_PER_TIMESTEP * max(len(text), 1))

    outputs = []
    for t in range(max_t):
        state = model.step(x, state, chars_onehot, char_lengths)
        pis, mu1, mu2, sigma1, sigma2, rho, es = model.output_params(state, bias_t)
        x = _sample_mdn(pis, mu1, mu2, sigma1, sigma2, rho, es)
        outputs.append(x[0].numpy())

        char_idx = int(state["phi"][0].argmax().item())
        is_eos = x[0, 2].item() >= 1.0
        if char_idx >= char_len - 1 and is_eos:
            break
        if char_idx >= char_len:
            break

    return np.stack(outputs, axis=0)


def strokes_to_lines(offsets):
    """Convert raw (dx, dy, eos) offsets into cleaned, aligned (x, y, eos) coordinates."""
    coords = drawing.offsets_to_coords(offsets)
    coords = drawing.denoise(coords)
    coords[:, :2] = drawing.align(coords[:, :2])
    return coords
