"""
fed_avg.py — Implémentation de l'algorithme FedAvg (Federated Averaging)
Stage : Détection de Fraude Bancaire avec Federated Learning

Référence: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.
"""

import numpy as np
import copy
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# 1. FedAvg pour modèles scikit-learn (LR, RF, XGBoost)
# via Soft Voting (agrégation des probabilités)
# ─────────────────────────────────────────────────────────────────────────────

class FederatedSoftVoting:
    """
    Fédération de modèles scikit-learn via soft voting.

    Chaque client entraîne un modèle local.
    La prédiction finale est la moyenne des probabilités de tous les clients.

    Note: Scikit-learn ne supporte pas la moyenne des poids comme les réseaux
    de neurones. On utilise donc le soft voting comme alternative pratique.
    """

    def __init__(self, base_model_fn, num_clients=25, clients_per_round=10,
                 num_rounds=5, random_state=42):
        """
        Args:
            base_model_fn: fonction qui retourne un modèle initialisé
                           ex: lambda: RandomForestClassifier(n_estimators=50)
            num_clients: nombre total de banques/clients
            clients_per_round: clients sélectionnés par round (C)
            num_rounds: nombre de rounds fédérés
            random_state: seed
        """
        self.base_model_fn      = base_model_fn
        self.num_clients        = num_clients
        self.clients_per_round  = clients_per_round
        self.num_rounds         = num_rounds
        self.random_state       = random_state
        self.client_models      = []
        self.history            = []

    def fit(self, clients_data, X_val=None, y_val=None, verbose=True):
        """
        Entraîne le système fédéré.

        Args:
            clients_data: list de (X_client, y_client) — un par client
            X_val, y_val: données de validation globales (optionnel)
        """
        from scripts.utils.metrics import evaluate_model, find_best_threshold

        np.random.seed(self.random_state)
        all_client_models = [None] * self.num_clients
        best_val_f1 = 0.0

        for round_num in range(1, self.num_rounds + 1):
            # Sélection aléatoire des clients participants
            selected = np.random.choice(
                self.num_clients, self.clients_per_round, replace=False)

            round_models = []
            for client_id in selected:
                X_c, y_c = clients_data[client_id]

                # Vérification : le client doit avoir les 2 classes
                if len(np.unique(y_c)) < 2:
                    if verbose:
                        print(f"  [Skipped] Client {client_id} n'a qu'une classe.")
                    continue

                model = self.base_model_fn()
                model.fit(X_c, y_c)
                all_client_models[client_id] = model
                round_models.append(model)

            self.client_models = [m for m in all_client_models if m is not None]

            if verbose:
                print(f"Round {round_num}/{self.num_rounds} — "
                      f"{len(round_models)} clients entraînés")

                if X_val is not None and y_val is not None and self.client_models:
                    y_prob = self.predict_proba(X_val)
                    thresh, f1 = find_best_threshold(y_val, y_prob)
                    print(f"  Val F1 (seuil={thresh:.2f}): {f1:.4f}")
                    self.history.append({'round': round_num, 'val_f1': f1})

                    if f1 > best_val_f1:
                        best_val_f1 = f1

        return self

    def predict_proba(self, X):
        """
        Prédit en agrégeant les probabilités de fraude de tous les clients.
        """
        if not self.client_models:
            raise RuntimeError("Le modèle fédéré n'a pas encore été entraîné.")

        probs = np.zeros(len(X))
        for model in self.client_models:
            probs += model.predict_proba(X)[:, 1]
        return probs / len(self.client_models)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_num_models(self):
        return len(self.client_models)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FedAvg pour réseaux de neurones PyTorch
# via agrégation des poids (vraie implémentation FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(global_model, client_models, client_sizes):
    """
    Agrégation FedAvg : moyenne pondérée des poids des clients.

    w_global = Σ (n_k / n_total) × w_k

    Args:
        global_model: modèle PyTorch global (state_dict sera mis à jour)
        client_models: list de modèles PyTorch (state_dicts des clients)
        client_sizes: list d'entiers (taille du dataset de chaque client)
    """
    import torch

    total_size = sum(client_sizes)
    global_dict = global_model.state_dict()

    # Initialiser les poids agrégés à zéro
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)

    # Somme pondérée
    for model, size in zip(client_models, client_sizes):
        client_dict = model.state_dict()
        weight = size / total_size
        for key in global_dict:
            global_dict[key] += client_dict[key].float() * weight

    global_model.load_state_dict(global_dict)
    return global_model


def client_train(model, dataloader, optimizer, criterion, device, epochs=1):
    """
    Entraîne un modèle PyTorch sur les données locales du client.

    Returns:
        float: perte moyenne sur le dernier epoch
    """
    model.train()
    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss = epoch_loss / max(len(dataloader), 1)
    return total_loss


def run_federated_training(global_model_fn, clients_data_loaders, client_sizes,
                           num_rounds=10, clients_per_round=None, lr=1e-3,
                           local_epochs=1, device=None, eval_fn=None, verbose=True):
    """
    Boucle complète d'entraînement fédéré FedAvg (PyTorch).

    Args:
        global_model_fn: callable qui retourne un modèle PyTorch initialisé
        clients_data_loaders: list de DataLoaders (un par client)
        client_sizes: list d'entiers (taille de chaque dataset client)
        num_rounds: nombre de rounds fédérés
        clients_per_round: nb de clients sélectionnés par round (défaut: tous)
        lr: learning rate
        local_epochs: epochs d'entraînement local par round
        device: torch.device
        eval_fn: fonction(model, round_num) -> dict de métriques (optionnel)

    Returns:
        (global_model, history)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_clients = len(clients_data_loaders)
    if clients_per_round is None:
        clients_per_round = num_clients

    global_model = global_model_fn().to(device)
    criterion = nn.BCELoss()
    history = []

    for round_num in range(1, num_rounds + 1):
        # Sélection des clients
        selected_ids = np.random.choice(num_clients, min(clients_per_round, num_clients),
                                        replace=False)

        client_models = []
        selected_sizes = []
        round_losses = []

        for client_id in selected_ids:
            # Copier le modèle global vers le client
            client_model = global_model_fn().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            optimizer = optim.Adam(client_model.parameters(), lr=lr)
            loss = client_train(client_model, clients_data_loaders[client_id],
                                optimizer, criterion, device, epochs=local_epochs)

            client_models.append(client_model)
            selected_sizes.append(client_sizes[client_id])
            round_losses.append(loss)

        # Agrégation FedAvg
        global_model = fedavg_aggregate(global_model, client_models, selected_sizes)

        round_info = {
            'round': round_num,
            'avg_loss': np.mean(round_losses),
            'clients_selected': len(selected_ids)
        }

        if eval_fn:
            metrics = eval_fn(global_model, round_num)
            round_info.update(metrics)

        history.append(round_info)

        if verbose:
            loss_str = f"{round_info['avg_loss']:.4f}"
            metric_str = ""
            if 'f1' in round_info:
                metric_str = f" | F1={round_info['f1']:.4f}"
            print(f"Round {round_num:02d}/{num_rounds} — "
                  f"Loss: {loss_str} | Clients: {len(selected_ids)}{metric_str}")

    return global_model, history
