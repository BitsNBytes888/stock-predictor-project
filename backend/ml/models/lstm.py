import torch
import torch.nn as nn
import numpy as np


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

        self.model = nn.Sequential(self.lstm, self.fc)

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

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
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

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            lstm_out, _ = self.lstm(X_tensor)
            last_hidden = lstm_out[:, -1, :]
            preds = self.fc(last_hidden)

        return preds.cpu().numpy().flatten()
