"""Non-connectome baselines.

These do not model the fly at all; they bound how hard the task is and, more
importantly, they quantify the shortcuts the connectome network must *not* be
using:

``area``
    A classifier that sees only the total ink.  On condition A it does well
    (dots get bigger/more numerous together); on the controlled conditions B and
    C it must collapse to chance, which is what makes those conditions valid
    tests of numerosity rather than brightness.

``pixels``
    Logistic regression on the raw 32x32 image -- the strongest purely
    feed-forward linear read of the stimulus.

``cnn``
    A small convolutional network.  This is an upper reference for task
    difficulty and is deliberately kept out of the connectome pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    macro_f1: float
    confusion: list
    extra: dict

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "confusion": self.confusion,
            **self.extra,
        }


def _metrics(y, pred, n_classes):
    from .metrics import classification_metrics

    return classification_metrics(y, pred, n_classes)


# --------------------------------------------------------------------------- #
def run_area_baseline(train: dict, tests: dict, n_classes: int, *, seed: int = 0):
    """Nearest-centroid on total ink only (no training needed, fully transparent)."""
    y = train["labels"]
    ink = train["ink"]
    classes = np.unique(y)
    means = np.array([ink[y == c].mean() for c in classes])
    out: dict[str, BaselineResult] = {}
    for mode, ds in tests.items():
        pred = classes[np.abs(ds["ink"][:, None] - means[None, :]).argmin(1)]
        m = _metrics(ds["labels"], pred, n_classes)
        out[mode] = BaselineResult(
            "area", m["accuracy"], m["macro_f1"], m["confusion"],
            {"ink_class_means": means.tolist(), "mode": mode},
        )
    return out


def run_pixel_baseline(
    train: dict, tests: dict, n_classes: int, *, seed: int = 0, C: float = 0.1
):
    """Logistic regression on the flattened image."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = train["images"].reshape(len(train["labels"]), -1)
    y = train["labels"]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=600, n_jobs=-1, random_state=seed)
    clf.fit(scaler.transform(X), y)
    out: dict[str, BaselineResult] = {}
    for mode, ds in tests.items():
        Xt = scaler.transform(ds["images"].reshape(len(ds["labels"]), -1))
        pred = clf.predict(Xt)
        m = _metrics(ds["labels"], pred, n_classes)
        out[mode] = BaselineResult(
            "pixels", m["accuracy"], m["macro_f1"], m["confusion"], {"mode": mode}
        )
    return out


def run_cnn_baseline(
    train: dict,
    tests: dict,
    n_classes: int,
    *,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-3,
    device: str = "cpu",
):
    """A deliberately small CNN: reference ceiling for task difficulty."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, n_classes)
            )

        def forward(self, x):
            return self.head(self.features(x))

    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    X = torch.as_tensor(train["images"], dtype=torch.float32).unsqueeze(1)
    y = torch.as_tensor(train["labels"], dtype=torch.long)
    n = len(y)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(device)
            yb = y[idx].to(device)
            loss = ce(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    net.eval()
    out: dict[str, BaselineResult] = {}
    with torch.no_grad():
        for mode, ds in tests.items():
            xt = torch.as_tensor(ds["images"], dtype=torch.float32).unsqueeze(1).to(device)
            pred = torch.cat(
                [net(xt[i : i + 512]).argmax(1).cpu() for i in range(0, len(xt), 512)]
            ).numpy()
            m = _metrics(ds["labels"], pred, n_classes)
            out[mode] = BaselineResult(
                "cnn", m["accuracy"], m["macro_f1"], m["confusion"], {"mode": mode}
            )
    return out
