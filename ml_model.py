# ml_model.py
from __future__ import annotations
from typing import List, Dict, Optional
import os
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Default paths
DEFAULT_MODEL_PATH = os.path.join("model", "phishing_model.pkl")
DEFAULT_DATASET_PATH = "dataset.csv"


def extract_features_from_url(url: str) -> List[float]:
    """Simple URL feature extractor (6 features)."""
    if not isinstance(url, str):
        url = str(url)
    has_https = 1 if url.startswith("https://") else 0
    parsed = urlparse(url)
    hostname = parsed.netloc or ""
    path = parsed.path or ""
    len_hostname = len(hostname)
    len_path = len(path)
    count_at = url.count("@")
    count_hyphen = url.count("-")
    count_digits = sum(ch.isdigit() for ch in url)
    return [has_https, len_hostname, len_path, count_at, count_hyphen, count_digits]


class PhishingDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        random_state: int = 42,
        use_grid_search: bool = False,
    ):
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.random_state = random_state
        self.use_grid_search = use_grid_search

    @staticmethod
    def default_dataset() -> pd.DataFrame:
        data = {
            "has_https":       [0, 1, 0, 1, 0, 1],
            "len_hostname":    [12, 8, 20, 6, 15, 9],
            "len_path":        [10, 1, 30, 0, 5, 2],
            "count_at":        [0, 0, 1, 0, 0, 0],
            "count_hyphen":    [1, 0, 2, 0, 1, 0],
            "count_digits":    [2, 0, 5, 0, 1, 0],
            "is_phishing":     [1, 0, 1, 0, 1, 0],
        }
        return pd.DataFrame(data)

    def _numeric_feature_filter(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Keep only numeric columns (convertible to numeric) except target."""
        possible_features = [c for c in df.columns if c != target]
        numeric_cols = []
        for c in possible_features:
            # try convert to numeric (without changing original until confirmed)
            coerced = pd.to_numeric(df[c], errors="coerce")
            # if not all NaN -> accept column (and will drop NaNs later)
            if coerced.notna().any():
                numeric_cols.append(c)
        if not numeric_cols:
            raise ValueError("No numeric feature columns found in dataset.")
        df2 = df[numeric_cols + [target]].copy()
        df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df2 = df2.dropna(axis=0)
        self.feature_names = numeric_cols
        return df2

    def load_dataset(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load or validate dataset.
        Supports:
         - dataset.csv with numeric feature columns + 'label' or 'is_phishing'
         - dataset.csv with 'url' and 'label'/'is_phishing' (will extract features)
        """
        if df is None:
            if os.path.exists(DEFAULT_DATASET_PATH):
                df = pd.read_csv(DEFAULT_DATASET_PATH)
            else:
                df = self.default_dataset()

        # Determine target column
        if "is_phishing" in df.columns:
            target = "is_phishing"
        elif "label" in df.columns:
            df = df.copy()
            df["is_phishing"] = df["label"]
            target = "is_phishing"
        else:
            # If only url present, we still need a label — otherwise error
            if "url" in df.columns:
                raise ValueError("Dataset has 'url' but no 'label' or 'is_phishing' column.")
            raise ValueError("Dataset must contain 'is_phishing' or 'label' target column.")

        # Case A: If 'url' column present, build features from URL
        if "url" in df.columns:
            df2 = df.copy()
            # compute features
            feats = df2["url"].apply(lambda u: extract_features_from_url(u))
            feats_df = pd.DataFrame(
                feats.tolist(),
                columns=["has_https","len_hostname","len_path","count_at","count_hyphen","count_digits"],
            )
            df_final = pd.concat([feats_df, df2[[target]].reset_index(drop=True)], axis=1)
            self.feature_names = ["has_https","len_hostname","len_path","count_at","count_hyphen","count_digits"]
            return df_final

        # Case B: numeric features already present -> filter numeric columns
        return self._numeric_feature_filter(df, target)

    def _build_pipeline(self) -> Pipeline:
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
        return pipeline

    def train_model(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        cross_val: bool = True,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        df = self.load_dataset(df)
        X = df[self.feature_names]
        y = df["is_phishing"]

        if len(df) < 10:
            cross_val = True

        # Try stratified split; fallback if it fails
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        self.pipeline = self._build_pipeline()

        if self.use_grid_search:
            param_grid = {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [None, 10],
            }
            cv = StratifiedKFold(n_splits=max(2, min(cv_folds, max(2, len(y_train)))))
            grid = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1)
            grid.fit(X_train, y_train)
            self.pipeline = grid.best_estimator_
        else:
            self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_proba = None
        try:
            if hasattr(self.pipeline, "predict_proba"):
                y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None and len(np.unique(y_test)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                pass

        if cross_val and len(df) >= cv_folds:
            try:
                scores = cross_val_score(self.pipeline, X, y, cv=cv_folds, scoring="f1", n_jobs=-1)
                metrics["cv_f1_mean"] = float(np.mean(scores))
                metrics["cv_f1_std"] = float(np.std(scores))
            except Exception:
                pass

        # Save model
        if self.model_path:
            self.save(self.model_path)

        return metrics

    def predict_phishing(self, url_features: List[float]) -> Dict[str, object]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded. Call train_model() or load(path).")
        if self.feature_names is None:
            raise RuntimeError("Feature names unknown. Train or load a model first.")
        if len(url_features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features: {self.feature_names}. Got {len(url_features)}.")
        X = np.array(url_features).reshape(1, -1)
        pred = self.pipeline.predict(X)[0]
        confidence = None
        if hasattr(self.pipeline, "predict_proba"):
            try:
                proba = self.pipeline.predict_proba(X)[0]
                confidence = float(np.max(proba))
            except Exception:
                confidence = None
        return {
            "is_phishing": bool(int(pred)),
            "confidence": confidence,
            "risk_level": "High" if int(pred) == 1 else "Low",
        }

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Nothing to save — model not trained.")
        payload = {"pipeline": self.pipeline, "feature_names": self.feature_names}
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(payload, path)

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise RuntimeError("Invalid model payload.")
        self.pipeline = payload["pipeline"]
        self.feature_names = payload.get("feature_names")


if __name__ == "__main__":
    detector = PhishingDetector(model_path=DEFAULT_MODEL_PATH, use_grid_search=False)
    if os.path.exists(DEFAULT_DATASET_PATH):
        print("Loading dataset:", DEFAULT_DATASET_PATH)
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        print("Training model...")
        metrics = detector.train_model(df=df, cross_val=True, cv_folds=5)
        print("Training finished. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("Saved model to:", detector.model_path)
    else:
        print("No dataset.csv found. Put dataset.csv in project root with either:")
        print("  (A) numeric feature columns + 'label' or 'is_phishing' OR")
        print("  (B) 'url' and 'label' (features will be extracted automatically).")
