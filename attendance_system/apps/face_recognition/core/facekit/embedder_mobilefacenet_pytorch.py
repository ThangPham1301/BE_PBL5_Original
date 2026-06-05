from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np


def lazy_import_cv2():
    import cv2  # type: ignore

    return cv2


def lazy_import_torch():
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    return torch, nn, F


class MobileFaceNetEmbedder:
    CFG = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]

    def __init__(self, emb: int = 512):
        torch, nn, _ = lazy_import_torch()
        super_cls = nn.Module

        class ConvBN(super_cls):
            def __init__(self, inp, oup, k=3, s=1, p=1, groups=1):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.PReLU(oup),
                )

            def forward(self, x):
                return self.block(x)

        class Bottleneck(super_cls):
            def __init__(self, inp, oup, stride, expand_ratio):
                super().__init__()
                hidden = inp * expand_ratio
                self.use_res = stride == 1 and inp == oup
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.PReLU(hidden),
                    nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.PReLU(hidden),
                    nn.Conv2d(hidden, oup, 1, bias=False),
                    nn.BatchNorm2d(oup),
                )

            def forward(self, x):
                return x + self.conv(x) if self.use_res else self.conv(x)

        class MobileFaceNet(super_cls):
            def __init__(self, embedding_size=512):
                super().__init__()
                self.stem = nn.Sequential(ConvBN(3, 64, s=2), ConvBN(64, 64, groups=64))
                layers, inp = [], 64
                for t, c, n, s in MobileFaceNetEmbedder.CFG:
                    for i in range(n):
                        layers.append(Bottleneck(inp, c, s if i == 0 else 1, t))
                        inp = c
                self.blocks = nn.Sequential(*layers)
                self.conv_last = ConvBN(128, 512, k=1, s=1, p=0)
                self.gdc = nn.Sequential(
                    nn.Conv2d(512, 512, 7, groups=512, bias=False),
                    nn.BatchNorm2d(512),
                )
                self.output = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512, embedding_size, bias=False),
                    nn.BatchNorm1d(embedding_size),
                )

            def forward(self, x):
                return self.output(self.gdc(self.conv_last(self.blocks(self.stem(x)))))

        self.torch = torch
        self.F = _
        self.model = MobileFaceNet(emb)


class MobileFaceNetPyTorchEmbedder:
    """PyTorch MobileFaceNet embedding-only model trained in recognition_pbl5.ipynb."""

    def __init__(self, model_path: Path, embedding_size: int = 512):
        self.model_path = Path(model_path)
        self.cv2 = lazy_import_cv2()
        self.torch, self.nn, self.F = lazy_import_torch()
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')

        wrapper = MobileFaceNetEmbedder(embedding_size)
        self.model = wrapper.model.to(self.device)
        state = self.torch.load(str(self.model_path), map_location=self.device)
        self.model.load_state_dict(self._extract_model_state(state), strict=True)
        self.model.eval()

    def _extract_model_state(self, checkpoint: Any) -> Dict[str, Any]:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']

        if not isinstance(checkpoint, dict):
            raise ValueError('Invalid PyTorch checkpoint format')

        cleaned = OrderedDict()
        for key, value in checkpoint.items():
            clean_key = key
            for prefix in ('module.', 'model.'):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            cleaned[clean_key] = value
        return cleaned

    def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = self.cv2.cvtColor(face_bgr, self.cv2.COLOR_BGR2RGB)
        face_rgb = self.cv2.resize(face_rgb, (112, 112), interpolation=self.cv2.INTER_LINEAR)
        face = face_rgb.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        tensor = self.torch.from_numpy(face).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            embedding = self.model(tensor).float()
            embedding = self.F.normalize(embedding, dim=1)

        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
