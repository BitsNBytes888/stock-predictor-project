import torch
import torch.nn as nn
import numpy as np

from backend.ml.preprocessing.scaling import StandardScaler3D


class LSTMModel:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        lr: float = 1e-3,
        epochs: int = 10,
        device: str | None = None,
    ):
        """
        input_dim: number of features per timestep
        hidden_dim: size of LSTM hidden state
        num_layers: stacked LSTM layers
        lr: learning rate
        epochs: training epochs per walk-forward step
        """

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._build_model()

    def _build_model(self):
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_dim, 1)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)

    # --- PyTorch plumbing ---
    def parameters(self):
        return list(self.lstm.parameters()) + list(self.fc.parameters())

    def to(self, device):
        self.lstm.to(device)
        self.fc.to(device)

    # --- Training ---
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (samples, seq_len, features)
        y: (samples,)
        """

        self.scaler = StandardScaler3D().fit(X)
        X_scaled = self.scaler.transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        y_tensor = y_tensor.view(-1, 1)

        for _ in range(self.epochs):
            self.optimizer.zero_grad()

            lstm_out, _ = self.lstm(X_tensor)
            last_hidden = lstm_out[:, -1, :]
            preds = self.fc(last_hidden)

            loss = self.criterion(preds, y_tensor)
            loss.backward()
            self.optimizer.step()

    # --- Prediction ---
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (batch, seq_len, features)
        Returns: (batch,)
        """

        self.lstm.eval()

        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            lstm_out, _ = self.lstm(X_tensor)
            last_hidden = lstm_out[:, -1, :]
            preds = self.fc(last_hidden)

        return preds.cpu().numpy().flatten()

    # --- Persistence ---
    def save(self, path: str) -> None:
        torch.save({
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "lr": self.lr,
                "epochs": self.epochs,
            },
            "lstm_state": self.lstm.state_dict(),
            "fc_state": self.fc.state_dict(),
            "scaler": self.scaler,
        }, path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**ckpt["config"])
        model.lstm.load_state_dict(ckpt["lstm_state"])
        model.fc.load_state_dict(ckpt["fc_state"])
        model.scaler = ckpt["scaler"]
        return model
