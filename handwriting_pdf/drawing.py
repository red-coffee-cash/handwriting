"""Stroke preprocessing/postprocessing utilities.

Ported from sjvasquez/handwriting-synthesis (drawing.py), which implements
the data conventions from Graves (2013), "Generating Sequences With
Recurrent Neural Networks" (https://arxiv.org/abs/1308.0850).
"""
from collections import defaultdict

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Character set the pretrained model was trained on (IAM-OnDB alphabet).
alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
]
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))

MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75


def encode_ascii(ascii_string):
    """Encode an ASCII string to an array of alphabet indices, null-terminated."""
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)) + [0])


def align(coords):
    """Corrects for global slant/offset in handwriting strokes."""
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords


def denoise(coords):
    """Smoothing filter to mitigate artifacts of the original data collection."""
    splits = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []
    for stroke in splits:
        if len(stroke) == 0:
            continue
        if len(stroke) >= 8:
            x_new = savgol_filter(stroke[:, 0], 7, 3, mode='nearest')
            y_new = savgol_filter(stroke[:, 1], 7, 3, mode='nearest')
        else:
            x_new = stroke[:, 0]
            y_new = stroke[:, 1]
        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])
        stroke = np.concatenate([xy_coords, stroke[:, 2].reshape(-1, 1)], axis=1)
        new_coords.append(stroke)

    if not new_coords:
        return coords
    return np.vstack(new_coords)


def offsets_to_coords(offsets):
    """Convert from (dx, dy, eos) offsets to cumulative (x, y, eos) coordinates."""
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def strokes_to_path_segments(offsets):
    """offsets: (N, 3) array of (dx, dy, eos). Returns list of point-lists,
    each representing one continuous pen-down stroke to draw as straight lines.

    Path construction follows the original reference implementation
    (sjvasquez/handwriting-synthesis demo.py `_draw`): straight-line segments
    only, broken into a new sub-path every time a point's end-of-stroke flag
    is set, plus a 1.5x coordinate scale applied before the denoise/align
    cleanup. Legibility comes from the density of points the RNN emits, not
    from curve fitting, so no bezier smoothing is applied here.
    """
    offsets = offsets.copy()
    offsets[:, :2] *= 1.5
    coords = offsets_to_coords(offsets)
    coords = denoise(coords)
    coords[:, :2] = align(coords[:, :2])

    segments = []
    current = []
    for x, y, eos in coords:
        current.append((x, y))
        if eos == 1:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments
