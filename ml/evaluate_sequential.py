"""
Phase 5 Batch 6: Sequential model CV pipeline — LSTM and GRU.

Standalone script; completely separate from the tabular CV pipeline in
ml/evaluate.py. Produces cv_metrics_{pos}_seq.csv files in the same schema
as the tabular pipeline for direct comparison.

Usage:
    python -m ml.evaluate_sequential                    # all positions, lstm + gru
    python -m ml.evaluate_sequential --position GK      # single position
    python -m ml.evaluate_sequential --model lstm       # one model only
    python -m ml.evaluate_sequential --epochs 30        # fewer epochs
    python -m ml.evaluate_sequential --compare          # print vs lgbm/ridge

Outputs:
    logs/training/cv_metrics_{pos}_seq.csv

Verification gate: lstm/gru must beat lgbm MAE on >= 2 positions before
TFT/N-BEATS are implemented.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ml.evaluate import CV_FOLDS, LOGS_DIR, get_feature_cols
from ml.features import VALID_POSITIONS, build_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

torch.manual_seed(42)

# -------------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------------

MAX_SEQ_LEN = 38   # max GWs per season (PL has 38 rounds)
HIDDEN_SIZE = 64
N_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 50


# -------------------------------------------------------------------------
# Sequence preparation
# -------------------------------------------------------------------------

def _compute_col_means(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Per-feature means over valid (non-padded, non-NaN) timesteps in X.
    X: (n_seqs, max_seq_len, n_features); mask: (n_seqs, max_seq_len) bool.
    """
    flat_x    = X.reshape(-1, X.shape[-1])
    flat_mask = mask.reshape(-1)
    valid     = flat_x[flat_mask]
    means     = np.nanmean(valid, axis=0)
    return np.where(np.isnan(means), 0.0, means).astype(np.float32)


def _fill_nans(arr: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Replace NaN entries in 2-D arr (n_rows, n_feat) with per-column means."""
    result = arr.copy()
    for j in range(arr.shape[1]):
        nan_rows = np.isnan(result[:, j])
        if nan_rows.any():
            result[nan_rows, j] = means[j]
    return result


def build_sequences(
    df: pd.DataFrame,
    feat_cols: list[str],
    col_means: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape flat feature matrix into padded player-season sequences.

    Sequences are ordered by (player_code, season_id) groupby, then gw.
    Sequences shorter than MAX_SEQ_LEN are zero-padded at the end.

    Parameters
    ----------
    df        : feature matrix — one row per (player_code, season_id, gw)
    feat_cols : feature column names
    col_means : per-feature means from training fold for NaN imputation;
                pass None to compute from this fold (useful only for training)

    Returns
    -------
    X    : (n_seqs, MAX_SEQ_LEN, n_features) float32
    y    : (n_seqs, MAX_SEQ_LEN) float32
    mask : (n_seqs, MAX_SEQ_LEN) bool — True for valid (non-padded) timesteps
    """
    n_feat  = len(feat_cols)
    groups  = df.groupby(['player_code', 'season_id'], sort=False)
    n_seqs  = groups.ngroups

    X    = np.zeros((n_seqs, MAX_SEQ_LEN, n_feat), dtype=np.float32)
    y    = np.zeros((n_seqs, MAX_SEQ_LEN),          dtype=np.float32)
    mask = np.zeros((n_seqs, MAX_SEQ_LEN),          dtype=bool)

    for seq_idx, (_, grp) in enumerate(groups):
        grp   = grp.sort_values('gw')
        t_len = min(len(grp), MAX_SEQ_LEN)

        x_raw = grp[feat_cols].values[:t_len].astype(np.float32)
        if col_means is not None:
            x_raw = _fill_nans(x_raw, col_means)
        else:
            x_raw = np.nan_to_num(x_raw, nan=0.0)

        X[seq_idx, :t_len] = x_raw
        y[seq_idx, :t_len] = grp['total_points'].values[:t_len].astype(np.float32)
        mask[seq_idx, :t_len] = True

    return X, y, mask


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.X    = torch.from_numpy(X)
        self.y    = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], self.mask[idx]


# -------------------------------------------------------------------------
# PyTorch models
# -------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        return self.head(out).squeeze(-1)  # (batch, seq_len)


class GRURegressor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


_MODEL_CLASSES: dict[str, type] = {
    'lstm': LSTMRegressor,
    'gru':  GRURegressor,
}


# -------------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_n    = 0
    for x_batch, y_batch, mask_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch)                             # (batch, seq_len)
        loss  = criterion(preds[mask_batch], y_batch[mask_batch])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n           = int(mask_batch.sum().item())
        total_loss += loss.item() * n
        total_n    += n
    return total_loss / total_n if total_n > 0 else float('nan')


@torch.no_grad()
def _get_flat_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (flat_preds, flat_targets) over all valid (non-padded) timesteps
    in loader order (sequences iterated in dataset order, no shuffle).
    """
    model.eval()
    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for x_batch, y_batch, mask_batch in loader:
        preds = model(x_batch)
        all_preds.append(preds[mask_batch].numpy())
        all_targets.append(y_batch[mask_batch].numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    position: str,
    fold: int,
) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)

    k          = min(10, len(y_true))
    top_actual = set(np.argsort(y_true)[-k:])
    top_pred   = set(np.argsort(y_pred)[-k:])
    top10      = len(top_actual & top_pred) / k

    return {
        'model':      model_name,
        'position':   position,
        'fold':       fold,
        'mae':        round(float(mae),   4),
        'rmse':       round(float(rmse),  4),
        'r2':         round(float(r2),    4),
        'spearman':   round(float(rho),   4),
        'top10_prec': round(float(top10), 4),
    }


# -------------------------------------------------------------------------
# CV loop
# -------------------------------------------------------------------------

def run_sequential_cv(
    position: str,
    model_names: list[str] | None = None,
    n_epochs: int = N_EPOCHS,
) -> pd.DataFrame:
    """
    Expanding-window CV for sequential models on one position.

    Uses the same CV_FOLDS as the tabular pipeline:
        Fold 1: train s7,      val s8
        Fold 2: train s7-8,    val s9
        Fold 3: train s7-9,    val s10

    Returns a DataFrame of per-fold metrics (same schema as cv_metrics_{pos}.csv).
    """
    if model_names is None:
        model_names = ['lstm', 'gru']

    df        = build_feature_matrix(position)
    feat_cols = get_feature_cols(df)
    n_feat    = len(feat_cols)

    records: list[dict] = []

    for fold_idx, (train_seasons, val_season) in enumerate(CV_FOLDS, start=1):
        train_df = df[df['season_id'].isin(train_seasons)].copy()
        val_df   = df[df['season_id'] == val_season].copy()

        if train_df.empty or val_df.empty:
            log.warning(f'[seq_cv] {position} fold {fold_idx}: empty split, skipping')
            continue

        log.info(
            f'[seq_cv] {position} fold {fold_idx} '
            f'(train s{train_seasons} -> val s{val_season}): '
            f'n_train={len(train_df):,} n_val={len(val_df):,}'
        )

        # --- Build sequences ---
        # First pass: raw training sequences to compute imputation statistics
        X_tr_raw, y_tr_raw, mask_tr_raw = build_sequences(train_df, feat_cols)
        col_means = _compute_col_means(X_tr_raw, mask_tr_raw)

        # Second pass: with NaN imputation applied
        X_tr,  y_tr,  mask_tr  = build_sequences(train_df, feat_cols, col_means)
        X_val, y_val, mask_val = build_sequences(val_df,   feat_cols, col_means)

        # --- Feature scaling (fit on training valid timesteps only) ---
        n_tr_seqs = X_tr.shape[0]
        n_val_seqs = X_val.shape[0]

        flat_tr      = X_tr.reshape(-1, n_feat)
        flat_mask_tr = mask_tr.reshape(-1)

        scaler = StandardScaler()
        scaler.fit(flat_tr[flat_mask_tr])

        X_tr_s  = scaler.transform(flat_tr).reshape(n_tr_seqs, MAX_SEQ_LEN, n_feat).astype(np.float32)
        X_val_s = scaler.transform(
            X_val.reshape(-1, n_feat)
        ).reshape(n_val_seqs, MAX_SEQ_LEN, n_feat).astype(np.float32)

        # Zero out padded positions after scaling (so LSTM sees clean zeros)
        X_tr_s[~mask_tr]   = 0.0
        X_val_s[~mask_val] = 0.0

        # --- Datasets and loaders ---
        train_loader = DataLoader(
            SequenceDataset(X_tr_s, y_tr, mask_tr),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            SequenceDataset(X_val_s, y_val, mask_val),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        for model_name in model_names:
            model     = _MODEL_CLASSES[model_name](n_feat)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.MSELoss()

            # --- Training ---
            for epoch in range(1, n_epochs + 1):
                train_loss = _train_epoch(model, train_loader, optimizer, criterion)
                if epoch % 10 == 0 or epoch == n_epochs:
                    log.info(
                        f'[seq_cv] {position}/{model_name} fold {fold_idx} '
                        f'epoch {epoch}/{n_epochs}: train_loss={train_loss:.4f}'
                    )

            # --- Evaluation ---
            y_pred, y_true = _get_flat_predictions(model, val_loader)
            metrics = _compute_metrics(y_true, y_pred, model_name, position, fold_idx)
            records.append(metrics)

            log.info(
                f'[seq_cv] {position}/{model_name} fold {fold_idx}: '
                f'MAE={metrics["mae"]:.4f}  RMSE={metrics["rmse"]:.4f}  '
                f'Spearman={metrics["spearman"]:.4f}'
            )

    return pd.DataFrame(records)


# -------------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------------

def _save_metrics(df: pd.DataFrame, position: str) -> None:
    out = LOGS_DIR / f'cv_metrics_{position}_seq.csv'
    df.to_csv(out, index=False)
    log.info(f'[save] {out}')


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('model')[['mae', 'rmse', 'r2', 'spearman', 'top10_prec']]
        .mean()
        .round(4)
    )


def _compare_with_tabular(position: str, seq_summary: pd.DataFrame) -> None:
    tab_path = LOGS_DIR / f'cv_metrics_{position}.csv'
    if not tab_path.exists():
        log.warning(f'No tabular CV metrics at {tab_path}; cannot compare')
        return

    tab = pd.read_csv(tab_path)
    tab_summary = tab.groupby('model')[['mae', 'rmse', 'spearman']].mean().round(4)

    ref_models = ['ridge', 'lgbm']
    ref_rows = tab_summary.loc[[m for m in ref_models if m in tab_summary.index]]

    print(f'\n--- {position}: sequential vs tabular (mean across folds) ---')
    print(
        pd.concat([
            ref_rows,
            seq_summary[['mae', 'rmse', 'spearman']],
        ]).to_string()
    )


def _gate_summary(
    all_results: dict[str, dict[str, float]],
    model_names: list[str],
    positions: list[str],
) -> None:
    """
    Print verification gate: sequential model beats LightGBM MAE on >= 2 positions.
    all_results[position][model_name] = mean_cv_mae
    """
    print(f'\n{"="*60}')
    print('Verification gate: sequential MAE < lgbm MAE')
    print('='*60)

    for model_name in model_names:
        n_pass   = 0
        n_total  = 0
        for pos in positions:
            if pos not in all_results:
                continue
            seq_mae  = all_results[pos].get(model_name)
            lgbm_mae = all_results[pos].get('lgbm_tabular')
            if seq_mae is None or lgbm_mae is None:
                continue
            n_total += 1
            if seq_mae < lgbm_mae:
                n_pass += 1
                log.info(f'  {model_name} {pos}: PASS (seq {seq_mae:.4f} < lgbm {lgbm_mae:.4f})')
            else:
                log.info(f'  {model_name} {pos}: FAIL (seq {seq_mae:.4f} >= lgbm {lgbm_mae:.4f})')

        gate_str = (
            f'PASS ({n_pass}/{n_total} positions beat lgbm)'
            if n_pass >= 2
            else f'FAIL ({n_pass}/{n_total} positions beat lgbm; need >= 2)'
        )
        print(f'  {model_name}: {gate_str}')

    tft_gate = any(
        sum(
            1 for pos in positions
            if all_results.get(pos, {}).get(m) is not None
            and all_results[pos].get('lgbm_tabular') is not None
            and all_results[pos][m] < all_results[pos]['lgbm_tabular']
        ) >= 2
        for m in model_names
    )
    print(
        f'\nTFT/N-BEATS gate (proceed only if above PASS): '
        f'{"PROCEED" if tft_gate else "HOLD"}'
    )
    print()


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL sequential model CV (LSTM/GRU)')
    p.add_argument(
        '--position', choices=list(VALID_POSITIONS) + ['all'],
        default='all', help='Position to evaluate (default: all)',
    )
    p.add_argument(
        '--model', choices=['lstm', 'gru', 'all'],
        default='all', help='Model to run (default: both)',
    )
    p.add_argument(
        '--epochs', type=int, default=N_EPOCHS,
        help=f'Max training epochs per fold (default: {N_EPOCHS})',
    )
    p.add_argument(
        '--compare', action='store_true',
        help='Print comparison with tabular lgbm/ridge after each position',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    positions   = list(VALID_POSITIONS) if args.position == 'all' else [args.position]
    model_names = ['lstm', 'gru']       if args.model == 'all'   else [args.model]

    # all_results[position][model_name or 'lgbm_tabular'] = mean_cv_mae
    all_results: dict[str, dict[str, float]] = {}

    for position in positions:
        log.info(f'\n{"="*60}\nPosition: {position}\n{"="*60}')

        cv_df = run_sequential_cv(position, model_names=model_names, n_epochs=args.epochs)

        if cv_df.empty:
            log.warning(f'[{position}] No CV results generated.')
            continue

        _save_metrics(cv_df, position)

        summary = _summarise(cv_df)
        print(f'\n--- {position} mean CV metrics (sequential) ---')
        print(summary.to_string())

        if args.compare:
            _compare_with_tabular(position, summary)

        # Collect MAE for gate check
        pos_results: dict[str, float] = {}
        for m in model_names:
            if m in summary.index:
                pos_results[m] = float(summary.loc[m, 'mae'])

        # Add lgbm tabular MAE for comparison
        tab_path = LOGS_DIR / f'cv_metrics_{position}.csv'
        if tab_path.exists():
            tab = pd.read_csv(tab_path)
            lgbm_rows = tab[tab['model'] == 'lgbm']
            if not lgbm_rows.empty:
                pos_results['lgbm_tabular'] = float(lgbm_rows['mae'].mean())

        all_results[position] = pos_results

    _gate_summary(all_results, model_names, positions)
