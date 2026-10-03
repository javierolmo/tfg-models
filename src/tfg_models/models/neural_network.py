"""PyTorch Tabular Neural Network with Entity Embeddings for property valuation."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tfg_models.core.trainer import BaseModelTrainer

logger = logging.getLogger(__name__)

# Limit PyTorch CPU threads to prevent thread pool contention/hangs on multi-core systems
if torch.get_num_threads() > 4:
    try:
        torch.set_num_threads(4)
    except Exception:
        pass


class TabularNNModule(nn.Module):
    """
    PyTorch Neural Network architecture combining Entity Embeddings for categorical
    features (postal_code) and multi-layer perceptron (MLP) for continuous variables.
    """

    def __init__(
        self,
        num_numerical: int,
        num_categories: int,
        embedding_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.embedding = nn.Embedding(
            num_embeddings=max(num_categories, 2),
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        in_dim = num_numerical + embedding_dim
        layers: List[nn.Module] = []

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass concatenating continuous features and category embeddings."""
        cat_embed = self.embedding(x_cat)
        combined = torch.cat([x_num, cat_embed], dim=1)
        return self.mlp(combined)


class PyTorchTabularRegressor:
    """
    Scikit-learn compatible regression wrapper around PyTorch TabularNNModule.
    Handles feature normalization, entity embedding mapping, and target scaling.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.15,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self.model: Optional[TabularNNModule] = None
        self.feature_names: List[str] = []
        self.postal_code_to_idx: Dict[str, int] = {}
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def _prepare_inputs(
        self, X: pd.DataFrame, is_train: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts and scales numeric features and maps categorical postal codes."""
        if is_train:
            self.feature_names = [col for col in X.columns if col != "postal_code"]
            X_num_raw = X[self.feature_names].values.astype(np.float32)
            X_num = self.scaler.fit_transform(X_num_raw)

            # Build vocabulary: index 0 reserved for unknown (<unk>)
            unique_pcs = sorted([str(pc) for pc in X["postal_code"].unique() if pd.notna(pc)])
            self.postal_code_to_idx = {pc: idx + 1 for idx, pc in enumerate(unique_pcs)}
        else:
            X_num_raw = X[self.feature_names].values.astype(np.float32)
            X_num = self.scaler.transform(X_num_raw)

        # Map postal codes (default to 0 for unknown)
        if "postal_code" in X.columns:
            pc_series = X["postal_code"].astype(str)
            X_cat = np.array(
                [self.postal_code_to_idx.get(pc, 0) for pc in pc_series],
                dtype=np.int64,
            )
        else:
            X_cat = np.zeros(len(X), dtype=np.int64)

        return X_num, X_cat

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PyTorchTabularRegressor":
        """Fits the neural network on tabular features and target prices."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = torch.device("cpu")

        X_num, X_cat = self._prepare_inputs(X, is_train=True)
        y_scaled = self.target_scaler.fit_transform(
            y.values.reshape(-1, 1).astype(np.float32)
        ).flatten()

        num_features = X_num.shape[1]
        num_categories = len(self.postal_code_to_idx) + 1  # +1 for unknown token

        self.model = TabularNNModule(
            num_numerical=num_features,
            num_categories=num_categories,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
        ).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_num).float(),
            torch.from_numpy(X_cat).long(),
            torch.from_numpy(y_scaled).float(),
        )

        effective_batch_size = min(self.batch_size, len(dataset))

        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for b_num, b_cat, b_y in loader:
                b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
                optimizer.zero_grad()
                preds = self.model(b_num, b_cat).squeeze(-1)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b_y)

            epoch_loss = total_loss / len(dataset)
            scheduler.step(epoch_loss)

        self.model.eval()
        self.model.to("cpu")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts property valuations in Euros (€)."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        self.model.eval()
        X_num, X_cat = self._prepare_inputs(X, is_train=False)

        with torch.no_grad():
            t_num = torch.from_numpy(X_num).float()
            t_cat = torch.from_numpy(X_cat).long()
            preds_scaled = self.model(t_num, t_cat).squeeze(-1).cpu().numpy()

        preds_scaled = np.atleast_1d(preds_scaled).reshape(-1, 1)
        preds_unscaled = self.target_scaler.inverse_transform(preds_scaled).flatten()

        # Prevent negative price outputs
        return np.clip(preds_unscaled, a_min=1000.0, a_max=None)


class NeuralNetworkTrainer(BaseModelTrainer):
    """
    PyTorch Tabular Neural Network trainer integrating entity embeddings
    and deep representations for real estate valuation.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(
            model_name="neural_network",
            encoding="categorical",
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> PyTorchTabularRegressor:
        """Trains the PyTorch Tabular Neural Network with entity embeddings."""
        logger.info(
            "Fitting PyTorch Tabular Neural Network (epochs=%d, batch_size=%d, lr=%.4f, emb_dim=%d)...",
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.embedding_dim,
        )
        model = PyTorchTabularRegressor(
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )
        return model.fit(X_train, y_train)

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates neural network performance and attaches architecture metadata."""
        report = super().evaluate(model, X_test, y_test)
        report["architecture"] = "Tabular MLP with Entity Embeddings"
        report["embedding_dim"] = self.embedding_dim
        report["hidden_dims"] = self.hidden_dims
        report["epochs"] = self.epochs
        report["loss_function"] = "HuberLoss"
        report["optimizer"] = "AdamW"
        report["postal_codes_learned"] = len(getattr(model, "postal_code_to_idx", {}))
        return report
