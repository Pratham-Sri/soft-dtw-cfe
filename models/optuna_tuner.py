"""
Optuna-based hyperparameter optimisation for the classifier.

Section 5.1: "Hyperparameters (dropout, learning rate, weight decay) are
optimised using Optuna [17], and training proceeds for up to 80 epochs."

Optimised hyperparameters:
  - dropout: [0.1, 0.5]
  - learning_rate: [1e-4, 1e-2] (log scale)
  - weight_decay: [1e-6, 1e-3] (log scale)
"""

import optuna
from optuna.trial import Trial

from soft_dtw_cfe.models.classifier import TSClassifier, train_classifier, evaluate_classifier
from soft_dtw_cfe.config import CLASSIFIER_CONFIG, DEVICE


def objective(trial: Trial, n_channels: int, n_classes: int, train_loader, test_loader):
    """Single Optuna trial: sample hyperparams, train, return val accuracy."""
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = TSClassifier(
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=dropout,
    )

    model, best_acc, _ = train_classifier(
        model, train_loader, test_loader,
        lr=lr,
        weight_decay=weight_decay,
        verbose=False,
    )

    return best_acc


def optimise_classifier(
    n_channels: int,
    n_classes: int,
    train_loader,
    test_loader,
    n_trials: int = None,
) -> dict:
    """
    Run Optuna hyperparameter search for the classifier.
    
    Args:
        n_channels: Number of input dimensions.
        n_classes: Number of classes.
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        n_trials: Number of Optuna trials (default from config).
    
    Returns:
        best_params: dict with 'dropout', 'lr', 'weight_decay'
    """
    if n_trials is None:
        n_trials = CLASSIFIER_CONFIG["optuna_n_trials"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, n_channels, n_classes, train_loader, test_loader),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"  [OPTUNA] Best params: {best_params}  (acc={study.best_value:.4f})")
    return best_params
