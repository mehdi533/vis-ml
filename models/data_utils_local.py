import joblib
import pandas as pd
import numpy as np
import torch
from typing import List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer


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


class Log1pRobustScaler:
    """Signed log1p transform followed by RobustScaler."""

    __slots__ = ("_robust",)

    def __init__(self):
        self._robust = RobustScaler()

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    @staticmethod
    def _signed_log1p(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.sign(x) * np.log1p(np.abs(x))

    @staticmethod
    def _signed_expm1(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.sign(x) * np.expm1(np.abs(x))

    def fit(self, X: np.ndarray, y=None):
        X2 = self._as_2d(X)
        Xt = self._signed_log1p(X2).astype(np.float32, copy=False)
        self._robust.fit(Xt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._as_2d(X)
        Xt = self._signed_log1p(X2).astype(np.float32, copy=False)
        return self._robust.transform(Xt).astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X2 = self._as_2d(X)
        Xt = self._signed_log1p(X2).astype(np.float32, copy=False)
        return self._robust.fit_transform(Xt).astype(np.float32, copy=False)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._as_2d(X)
        Xt = self._robust.inverse_transform(X2)
        return self._signed_expm1(Xt).astype(np.float32, copy=False)


class IdentityScaler:
    """No-op scaler for constant/binary features."""

    __slots__ = ()

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


class ColumnwiseScaler:
    """Apply a list of scalers column-by-column."""

    __slots__ = ("scalers_", "columns_")

    def __init__(self, scalers, columns):
        self.scalers_ = list(scalers)
        self.columns_ = list(columns)

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    def fit(self, X: np.ndarray, y=None):
        X2 = self._as_2d(X)
        for idx, scaler in enumerate(self.scalers_):
            scaler.fit(X2[:, [idx]])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._as_2d(X)
        out = np.empty_like(X2, dtype=np.float32)
        for idx, scaler in enumerate(self.scalers_):
            out[:, idx : idx + 1] = scaler.transform(X2[:, [idx]])
        return out

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X2 = self._as_2d(X)
        out = np.empty_like(X2, dtype=np.float32)
        for idx, scaler in enumerate(self.scalers_):
            out[:, idx : idx + 1] = scaler.fit_transform(X2[:, [idx]])
        return out

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._as_2d(X)
        out = np.empty_like(X2, dtype=np.float32)
        for idx, scaler in enumerate(self.scalers_):
            out[:, idx : idx + 1] = scaler.inverse_transform(X2[:, [idx]])
        return out


SCALER_FACTORY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "log1p": Log1pScaler,
}

RECOMMENDED_SCALER_FACTORY = {
    "standardscaler": StandardScaler,
    "minmaxscaler": MinMaxScaler,
    "robustscaler": RobustScaler,
    "log1p + standardscaler": Log1pScaler,
    "log1p+standardscaler": Log1pScaler,
    "log1p + robustscaler": Log1pRobustScaler,
    "log1p+robustscaler": Log1pRobustScaler,
    "powertransformer (yeo-johnson)": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
    "powertransformer(yeo-johnson)": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
    "powertransformer": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
    "binary (no scaling)": IdentityScaler,
    "constant (skip)": IdentityScaler,
}


def load_dataset(
    csv_path: str,
    target_cols: Optional[Sequence[str]] = None,
    remove_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(csv_path)

    targets = list(target_cols) if target_cols is not None else ["max_rocof", "f_nadir", "f_max"]
    drops = list(remove_cols) if remove_cols is not None else []

    missing = [c for c in targets + drops if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    feature_cols = [c for c in df.columns if c not in set(targets + drops)]
    if not feature_cols:
        raise ValueError("No feature columns left after removing targets/drops.")

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


def _resolve_feature_column(df: pd.DataFrame) -> str:
    if "feature" in df.columns:
        return "feature"
    if "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    return df.columns[0]


def _build_recommended_scaler(name: Optional[str], default_scaler_type: str):
    if not name or not isinstance(name, str):
        return SCALER_FACTORY.get(default_scaler_type, StandardScaler)()
    key = name.strip().lower()
    scaler_factory = RECOMMENDED_SCALER_FACTORY.get(key)
    if scaler_factory is None:
        return SCALER_FACTORY.get(default_scaler_type, StandardScaler)()
    return scaler_factory()


def load_scaler_recommendations(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    feature_col = _resolve_feature_column(df)
    if "recommended_scaler" not in df.columns:
        raise ValueError(f"Missing 'recommended_scaler' column in {csv_path}")
    return {
        str(feat): str(rec)
        for feat, rec in zip(df[feature_col], df["recommended_scaler"])
    }


def scale_data_with_recommendations(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    x_scaler_csv: str,
    y_scaler_csv: str,
    x_scaler_path="x_scaler.pkl",
    y_scaler_path="y_scaler.pkl",
    default_scaler_type="minmax",
    save_scaler=True,
):
    x_recos = load_scaler_recommendations(x_scaler_csv)
    y_recos = load_scaler_recommendations(y_scaler_csv)

    x_scalers = [
        _build_recommended_scaler(x_recos.get(col), default_scaler_type)
        for col in feature_cols
    ]
    y_scalers = [
        _build_recommended_scaler(y_recos.get(col), default_scaler_type)
        for col in target_cols
    ]

    x_scaler = ColumnwiseScaler(x_scalers, feature_cols)
    y_scaler = ColumnwiseScaler(y_scalers, target_cols)

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
