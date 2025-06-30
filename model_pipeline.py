import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, recall_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import optuna

all_y_true = []
all_y_pred = []

class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class CNN1D(nn.Module):
    def __init__(self, input_length=2048, output_dim=1, task="regression"):
        super(CNN1D, self).__init__()
        self.task = task
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3)

        conv_out_length = input_length
        for k in [5, 3, 3, 3, 3]:
            conv_out_length = (conv_out_length - (k - 1)) // 2
        self.flatten_dim = conv_out_length * 128

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.output = nn.Linear(512, output_dim if task == "classification" else 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.output(x).squeeze(1) if self.task == "regression" else self.output(x)

def extract_segment_features(xrf_array, num_segments=64):
    segment_length = xrf_array.shape[1] // num_segments
    features = []
    for i in range(num_segments):
        segment = xrf_array[:, i * segment_length: (i + 1) * segment_length]
        mean = segment.mean(axis=1)
        max_ = segment.max(axis=1)
        # std = segment.std(axis=1)
        # features.append(np.stack([mean, max_, std], axis=1))
        features.append(np.stack([mean, max_], axis=1))
    return np.concatenate(features, axis=1)

def evaluate_models_with_kfold(
    data_path,
    label_column,
    xrf_columns,
    metal_columns,
    task="classification",
    n_splits=5,
    latent_dim=32,
    num_epochs=100,
    random_state=42,
    device='cpu',
    preprocessing="none",
    use_optuna=False,
    input_methods=None,
    models=None,
    conf_matrix_condition=None,
    output_confusion_matrix=False,
):
    global all_y_true, all_y_pred
    all_y_true = []
    all_y_pred = []

    df = pd.read_csv(data_path)
    xrf = df[xrf_columns].values.astype(np.float32)
    metal = df[metal_columns].values.astype(np.float32)
    y_raw = df[label_column].values
    groups = df["ObjectID"].values

    if task == "regression":
        y = y_raw.astype(np.float32)
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)

    splitter = GroupKFold(n_splits=n_splits)
    split_iter = splitter.split(df, y, groups=groups)

    def preprocess(x):
        if preprocessing == "sqrt":
            return np.sqrt(np.clip(x, 0, None))
        elif preprocessing == "log":
            return np.log1p(np.clip(x, 0, None))
        return x.copy()

    def preprocess_and_scale(X, preprocessing, scaler_name):
        X = X.copy()
        if preprocessing == "log":
            X = np.log1p(np.clip(X, 0, None))
        elif preprocessing == "sqrt":
            X = np.sqrt(np.clip(X, 0, None))

        if scaler_name == "standard":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        return X

    results = []

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        print(f"Fold {fold + 1}/{n_splits}")

        xrf_train, xrf_test = xrf[train_idx], xrf[test_idx]
        metal_train, metal_test = metal[train_idx], metal[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Autoencoder
        xrf_train_ae = preprocess(xrf_train)
        xrf_test_ae = preprocess(xrf_test)
        ae = Autoencoder(input_dim=xrf_train.shape[1], latent_dim=latent_dim).to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        x_train_tensor = torch.tensor(xrf_train_ae, dtype=torch.float32).to(device)
        ae.train()
        for _ in range(num_epochs):
            opt.zero_grad()
            recon, _ = ae(x_train_tensor)
            loss = loss_fn(recon, x_train_tensor)
            loss.backward()
            opt.step()

        ae.eval()
        with torch.no_grad():
            x_train_tensor = torch.tensor(xrf_train_ae, dtype=torch.float32).to(device)
            x_test_tensor = torch.tensor(xrf_test_ae, dtype=torch.float32).to(device)
            _, latent_train = ae(x_train_tensor)
            _, latent_test = ae(x_test_tensor)
        latent_train = latent_train.cpu().numpy()
        latent_test = latent_test.cpu().numpy()

        xrf_seg_train = extract_segment_features(xrf_train)
        xrf_seg_test = extract_segment_features(xrf_test)

        all_inputs = {
            "metal_only": (metal_train, metal_test),
            "xrf_only": (xrf_train, xrf_test),
            "latent_only": (latent_train, latent_test),
            "feature": (xrf_seg_train, xrf_seg_test)
        }

        if task == "regression" and "metal+PLS" in (input_methods or []):
            all_inputs["metal+PLS"] = (metal_train, metal_test)
        if "xrf+CNN" in (input_methods or []):
            all_inputs["xrf+CNN"] = (xrf_train, xrf_test)

        for input_name, (X_train, X_test) in all_inputs.items():
            if input_methods and input_name not in input_methods:
                continue
            if input_name == "metal+PLS":
                allowed_models = ["PLSRegression"]
            elif input_name == "xrf+CNN":
                allowed_models = ["CNN"]
            else:
                allowed_models = models or []
            for model_name in allowed_models:
                if model_name == "PLSRegression" and (input_name != "metal+PLS" or task != "regression"):
                    continue

                if model_name == "PLSRegression":
                    model = PLSRegression(n_components=min(X_train.shape[1], 10))
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test).ravel()
                    results.append({
                        "fold": fold + 1, "method": input_name, "model": model_name,
                        "r2": r2_score(y_test, y_pred),
                        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "mae": mean_absolute_error(y_test, y_pred)
                    })
                    continue

                if model_name == "CNN":
                    x_train_cnn = preprocess_and_scale(X_train, preprocessing, scaler_name="standard")
                    x_test_cnn = preprocess_and_scale(X_test, preprocessing, scaler_name="standard")

                    x_train_tensor = torch.tensor(x_train_cnn[:, np.newaxis, :], dtype=torch.float32).to(device)
                    x_test_tensor = torch.tensor(x_test_cnn[:, np.newaxis, :], dtype=torch.float32).to(device)
                    y_train_tensor = torch.tensor(y_train, dtype=torch.long if task == "classification" else torch.float32).to(device)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.long if task == "classification" else torch.float32).to(device)

                    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
                    test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=32)

                    model = CNN1D(input_length=x_train_cnn.shape[1], output_dim=len(np.unique(y)) if task == "classification" else 1, task=task).to(device)
                    criterion = FocalLoss(alpha=1.0, gamma=2.0) if task == "classification" else nn.L1Loss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

                    for _ in range(8):
                        model.train()
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            optimizer.zero_grad()
                            out = model(xb)
                            loss = criterion(out, yb)
                            loss.backward()
                            optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        preds = model(x_test_tensor)
                        if task == "classification":
                            y_pred = preds.argmax(dim=1).cpu().numpy()
                            results.append({
                                "fold": fold + 1, "method": input_name, "model": model_name,
                                "acc": accuracy_score(y_test, y_pred),
                                "f1": f1_score(y_test, y_pred, average="macro"),
                                "recall": recall_score(y_test, y_pred, average="macro")
                            })
                            if output_confusion_matrix and conf_matrix_condition:
                                if input_name == conf_matrix_condition.get("method") and model_name == conf_matrix_condition.get("model"):
                                    all_y_true.extend(y_test)
                                    all_y_pred.extend(y_pred)
                        else:
                            y_pred = preds.cpu().numpy().flatten()
                            results.append({
                                "fold": fold + 1, "method": input_name, "model": model_name,
                                "r2": r2_score(y_test, y_pred),
                                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                                "mae": mean_absolute_error(y_test, y_pred)
                            })
                    continue                

                # with or without optuna
                def train_and_eval(model):
                    model.fit(X_train, y_train)
                    return model.predict(X_test)

                def objective(trial, algo):
                    if algo == "RandomForest":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                            "max_depth": trial.suggest_int("max_depth", 3, 12),
                            "random_state": random_state
                        }
                        model = RandomForestRegressor(**params) if task == "regression" else RandomForestClassifier(**params, class_weight='balanced')
                    elif algo == "XGBoost":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                            "max_depth": trial.suggest_int("max_depth", 3, 12),
                            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                            "random_state": random_state
                        }
                        model = XGBRegressor(**params) if task == "regression" else XGBClassifier(**params)
                    elif algo == "LightGBM":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                            "max_depth": trial.suggest_int("max_depth", 3, 12),
                            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                            "random_state": random_state
                        }
                        model = LGBMRegressor(**params) if task == "regression" else LGBMClassifier(**params, class_weight='balanced')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if task == "regression":
                        return mean_squared_error(y_test, y_pred)
                    return 1 - f1_score(y_test, y_pred, average="macro")

                if use_optuna:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(lambda trial: objective(trial, model_name), n_trials=30, timeout=300)
                    best_params = study.best_params
                    if model_name == "RandomForest":
                        model = RandomForestRegressor(**best_params) if task == "regression" else RandomForestClassifier(**best_params, class_weight='balanced')
                    elif model_name == "XGBoost":
                        model = XGBRegressor(**best_params) if task == "regression" else XGBClassifier(**best_params)
                    elif model_name == "LightGBM":
                        model = LGBMRegressor(**best_params) if task == "regression" else LGBMClassifier(**best_params, class_weight='balanced')
                else:
                    if model_name == "RandomForest":
                        model = RandomForestRegressor(n_estimators=100, random_state=random_state) if task == "regression" else RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
                    elif model_name == "XGBoost":
                        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=random_state) if task == "regression" else XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=random_state)
                    elif model_name == "LightGBM":
                        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=random_state, verbose=-1) if task == "regression" else LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=random_state, class_weight='balanced', verbose=-1)

                y_pred = train_and_eval(model)
                if task == "regression":
                    results.append({
                        "fold": fold + 1, "method": input_name, "model": model_name,
                        "r2": r2_score(y_test, y_pred),
                        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "mae": mean_absolute_error(y_test, y_pred)
                    })
                else:
                    results.append({
                        "fold": fold + 1, "method": input_name, "model": model_name,
                        "acc": accuracy_score(y_test, y_pred),
                        "f1": f1_score(y_test, y_pred, average="macro"),
                        "recall": recall_score(y_test, y_pred, average="macro")
                    })
                    if output_confusion_matrix and conf_matrix_condition:
                        if input_name == conf_matrix_condition.get("method") and model_name == conf_matrix_condition.get("model"):
                            all_y_true.extend(y_test)
                            all_y_pred.extend(y_pred)

    return pd.DataFrame(results)
