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


class MobileFaceNetFactory:
    """MobileFaceNet architecture used by recognition_pbl5_128d and bm6.pth."""

    def __init__(self, embedding_size: int = 128):
        torch, nn, F = lazy_import_torch()
        module_cls = nn.Module

        class ConvBN(module_cls):
            def __init__(self, inp, oup, k=3, s=1, p=1, groups=1):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.PReLU(oup),
                )

            def forward(self, x):
                return self.block(x)

        class ConvBlock(module_cls):
            def __init__(self, inp, oup, k=3, s=1, p=1, groups=1, linear=False):
                super().__init__()
                self.linear = linear
                self.conv = nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False)
                self.bn = nn.BatchNorm2d(oup)
                if not linear:
                    self.prelu = nn.PReLU(oup)

            def forward(self, x):
                x = self.bn(self.conv(x))
                return x if self.linear else self.prelu(x)

        class Bottleneck(module_cls):
            def __init__(self, inp, oup, stride, expand_ratio):
                super().__init__()
                hidden = inp * expand_ratio
                self.use_res = stride == 1 and inp == oup
                self.conv = nn.Sequential(
                    ConvBlock(inp, hidden, k=1, s=1, p=0),
                    ConvBlock(hidden, hidden, k=3, s=stride, p=1, groups=hidden),
                    ConvBlock(hidden, oup, k=1, s=1, p=0, linear=True),
                )

            def forward(self, x):
                return x + self.conv(x) if self.use_res else self.conv(x)

        class MobileFaceNet(module_cls):
            cfg = [
                (2, 64, 5, 2),
                (4, 128, 1, 2),
                (2, 128, 6, 1),
                (4, 128, 1, 2),
                (2, 128, 2, 1),
            ]

            def __init__(self, emb=128):
                super().__init__()
                self.conv1 = ConvBN(3, 64, k=3, s=2, p=1)
                self.dw_conv = ConvBN(64, 64, k=3, s=1, p=1, groups=64)

                layers = []
                inp = 64
                for t, c, n, s in self.cfg:
                    for i in range(n):
                        layers.append(Bottleneck(inp, c, stride=s if i == 0 else 1, expand_ratio=t))
                        inp = c
                self.blocks = nn.Sequential(*layers)

                self.conv2 = ConvBN(128, 512, k=1, s=1, p=0)
                self.gdc = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=(7, 6), groups=512, bias=False),
                    nn.BatchNorm2d(512),
                )
                self.linear1 = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512, emb, bias=False),
                    nn.BatchNorm1d(emb),
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.dw_conv(x)
                x = self.blocks(x)
                x = self.conv2(x)
                x = self.gdc(x)
                return self.linear1(x)

        self.torch = torch
        self.F = F
        self.model = MobileFaceNet(embedding_size)


class MobileFaceNetPyTorchEmbedder:
    """PyTorch MobileFaceNet embedder for bm6.pth."""

    def __init__(self, model_path: Path, embedding_size: int = 128):
        self.model_path = Path(model_path)
        self.embedding_size = int(embedding_size)
        self.input_height = 112
        self.input_width = 96
        self.cv2 = lazy_import_cv2()
        self.torch, self.nn, self.F = lazy_import_torch()
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

        wrapper = MobileFaceNetFactory(self.embedding_size)
        self.model = wrapper.model.to(self.device)
        state = self.torch.load(str(self.model_path), map_location=self.device)
        self.model.load_state_dict(self._extract_model_state(state), strict=True)
        self.model.eval()

    def _extract_model_state(self, checkpoint: Any) -> Dict[str, Any]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid PyTorch checkpoint format")

        cleaned = OrderedDict()
        for key, value in checkpoint.items():
            clean_key = key
            for prefix in ("module.", "model."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            cleaned[clean_key] = value
        return cleaned

    def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = self.cv2.cvtColor(face_bgr, self.cv2.COLOR_BGR2RGB)
        face_rgb = self.cv2.resize(face_rgb, (self.input_width, self.input_height), interpolation=self.cv2.INTER_LINEAR)
        face = face_rgb.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        tensor = self.torch.from_numpy(face).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            embedding = self.model(tensor).float()
            embedding = self.F.normalize(embedding, dim=1)

        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
