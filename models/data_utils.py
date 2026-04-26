# data_utils.py
# Dataset loading, filtering, scaling, and DataLoader helpers for model workflows.

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Scalers
# -----------------------------

class Log1pScaler:
    """Signed log1p transform followed by per-feature standardization."""

    __slots__ = ("mean_", "scale_")

    @staticmethod
    def _signed_log1p(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.sign(x) * np.log1p(np.abs(x))

    @staticmethod
    def _signed_expm1(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.sign(x) * np.expm1(np.abs(x))

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    def fit(self, X: np.ndarray, y=None):
        X2 = self._as_2d(X)
        Xt = self._signed_log1p(X2).astype(np.float32, copy=False)
        mean = Xt.mean(axis=0, dtype=np.float64)
        var = Xt.var(axis=0, dtype=np.float64)
        scale = np.sqrt(var)
        scale[scale == 0.0] = 1.0
        self.mean_ = mean.astype(np.float32, copy=False)
        self.scale_ = scale.astype(np.float32, copy=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._as_2d(X)
        Xt = self._signed_log1p(X2).astype(np.float32, copy=False)
        return (Xt - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X2 = self._as_2d(X)
        Xt = self._signed_log1p(X2).astype(np.float32, copy=False)
        mean = Xt.mean(axis=0, dtype=np.float64)
        var = Xt.var(axis=0, dtype=np.float64)
        scale = np.sqrt(var)
        scale[scale == 0.0] = 1.0
        self.mean_ = mean.astype(np.float32, copy=False)
        self.scale_ = scale.astype(np.float32, copy=False)
        return (Xt - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._as_2d(X)
        Xt = X2.astype(np.float32, copy=False) * self.scale_ + self.mean_
        return self._signed_expm1(Xt).astype(np.float32, copy=False)


SCALER_FACTORY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "log1p": Log1pScaler,
}

SCALER_CATALOG = sorted(SCALER_FACTORY.keys())


# -----------------------------
# Dataset / split helpers
# -----------------------------

def list_scalers():
    return list(SCALER_CATALOG)


def load_dataset(
    csv_path: str,
    target_cols: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    allowed_feature_cols: Optional[Sequence[str]] = None,
    allowed_feature_prefixes: Optional[Sequence[str]] = None,
    remove_cols: Optional[Sequence[str]] = None,
    remove_prefixes: Optional[Sequence[str]] = None,
    ignore_missing_remove_cols: bool = False,
    missing_fill_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(csv_path, low_memory=False)

    targets = list(target_cols) if target_cols is not None else ["max_rocof", "f_nadir", "f_max"]
    drops = list(remove_cols) if remove_cols is not None else []
    drop_prefixes = [str(prefix) for prefix in remove_prefixes] if remove_prefixes is not None else []
    allowed_cols = (
        {str(col) for col in allowed_feature_cols}
        if allowed_feature_cols is not None
        else None
    )
    allowed_prefixes = (
        [str(prefix) for prefix in allowed_feature_prefixes]
        if allowed_feature_prefixes is not None
        else []
    )

    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in CSV: {missing_targets}")

    if ignore_missing_remove_cols:
        drops = [c for c in drops if c in df.columns]
    else:
        missing_drops = [c for c in drops if c not in df.columns]
        if missing_drops:
            raise ValueError(f"Missing remove_cols in CSV: {missing_drops}")

    blocked = set(targets + drops)
    explicit_feature_cols = list(feature_cols) if feature_cols is not None else None
    if explicit_feature_cols is not None:
        missing_features = [c for c in explicit_feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing explicit feature_cols in CSV: {missing_features}")
        blocked_features = [c for c in explicit_feature_cols if c in blocked]
        if blocked_features:
            raise ValueError(
                f"Explicit feature_cols cannot overlap with targets/remove_cols: {blocked_features}"
            )
        blocked_by_prefix = [
            c
            for c in explicit_feature_cols
            if any(str(c).startswith(prefix) for prefix in drop_prefixes)
        ]
        if blocked_by_prefix:
            raise ValueError(
                f"Explicit feature_cols cannot match remove_prefixes: {blocked_by_prefix}"
            )
        feature_cols = explicit_feature_cols
    else:
        feature_cols = [
            c
            for c in df.columns
            if c not in blocked
            and not any(str(c).startswith(prefix) for prefix in drop_prefixes)
            and (
                allowed_cols is None
                and not allowed_prefixes
                or c in allowed_cols
                or any(str(c).startswith(prefix) for prefix in allowed_prefixes)
            )
        ]
    if not feature_cols:
        raise ValueError("No feature columns left after removing targets/drops.")

    # Optional NaN fill for training convenience (e.g., fill with -1.0).
    if missing_fill_value is not None:
        cols_to_fill = feature_cols + targets
        df[cols_to_fill] = df[cols_to_fill].fillna(float(missing_fill_value))

    X = df[feature_cols].values.astype(np.float32)
    y = df[targets].values.astype(np.float32)

    return X, y, feature_cols, targets


def split_data(X, y, test_size=0.3, val_fraction=0.5, random_state=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------
# Scaling workflows
# -----------------------------

def scale_data(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    x_scaler_path="x_scaler.pkl",
    y_scaler_path="y_scaler.pkl",
    scaler_type="minmax",
    save_scaler=True,
):
    x_scaler = SCALER_FACTORY[scaler_type]()
    y_scaler = SCALER_FACTORY[scaler_type]()

    X_train_norm = x_scaler.fit_transform(X_train)
    X_val_norm = x_scaler.transform(X_val)
    X_test_norm = x_scaler.transform(X_test)

    y_train_norm = y_scaler.fit_transform(y_train)
    y_val_norm = y_scaler.transform(y_val)
    y_test_norm = y_scaler.transform(y_test)

    if save_scaler:
        joblib.dump(x_scaler, x_scaler_path)
        joblib.dump(y_scaler, y_scaler_path)

    return (
        X_train_norm, X_val_norm, X_test_norm,
        y_train_norm, y_val_norm, y_test_norm,
        y_scaler,
    )

# -----------------------------
# Torch DataLoaders
# -----------------------------

def make_dataloaders(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    batch_size_train=16,
    batch_size_eval=128,
):
    def _to_tensor(array):
        return torch.as_tensor(array, dtype=torch.float32)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    val_ds = TensorDataset(_to_tensor(X_val), _to_tensor(y_val))
    test_ds = TensorDataset(_to_tensor(X_test), _to_tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size_eval, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size_eval, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
