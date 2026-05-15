"""
train_lora_lstm.py — CNN-LSTM training script for LoRa attack detection
Run this on Windows or a desktop machine after copying the JSONL files
from the Pi. No training happens on the Pi.

Input files (copy from Pi):
    lora_normal_data.jsonl
    lora_attack_data.jsonl

Output files:
    lora_attack_detector.h5
    lora_scaler.pkl

Setup:
    pip install tensorflow scikit-learn pandas numpy matplotlib joblib

Run:
    python train_lora_lstm.py
    python train_lora_lstm.py --time-steps 20 --threshold 0.5
"""

import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     Conv1D, MaxPooling1D,
                                     BatchNormalization)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
TIME_STEPS       = 20
EPOCHS           = 100
BATCH_SIZE       = 64
TRAIN_SPLIT      = 0.8
THRESHOLD        = 0.35      # back to 0.5 — let the sweep guide the final value
MODEL_OUT        = 'lora_attack_detector.h5'
SCALER_OUT       = 'lora_scaler.pkl'
FEATURES         = ['delta_t', 'abs_delta_t', 'hw_rssi', 'hw_snr', 'ipd', 'ipd_ratio']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(filepath, expected_label=None):
    rows = []
    skipped = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                hw_rssi = raw['physical_layer']['hw_rssi']
                if hw_rssi == -100:
                    skipped += 1
                    continue
                row = {
                    'seq':           raw['metadata']['seq'],
                    'arrival_local': raw['timestamps']['arrival_local'],
                    'delta_t':       raw['timestamps']['delta_t'],
                    'hw_rssi':       hw_rssi,
                    'hw_snr':        raw['physical_layer']['hw_snr'],
                    'ipd':           raw.get('ipd', 0.0),
                    'label':         raw['label'],
                }
                if expected_label is not None and row['label'] != expected_label:
                    skipped += 1
                    continue
                rows.append(row)
            except Exception:
                skipped += 1
    print(f"  Loaded {len(rows):,} records, skipped {skipped} from {filepath}")
    return pd.DataFrame(rows)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    """
    abs_delta_t  — collapses positive burst (+512) and negative stale (-128)
                   attacks into a single large value, easier for the model to learn
    ipd_ratio    — current IPD vs rolling median, burst attacks drop this near zero
    """
    df = df.copy()
    df['abs_delta_t'] = df['delta_t'].abs()
    median_ipd = df['ipd'].rolling(window=5, min_periods=1).median()
    df['ipd_ratio'] = df['ipd'] / (median_ipd + 0.01)
    return df


# ── Sequence windowing ────────────────────────────────────────────────────────

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_shape):
    """
    Simplified back to 2 LSTM layers — the 3-layer model needed more data
    and caused the model to collapse to always predicting attack.
    Conv1D catches burst patterns, LSTM layers catch temporal drift.
    """
    model = Sequential([
        # Local pattern detection
        Conv1D(filters=64, kernel_size=3, activation='relu',
               padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Temporal pattern — catches short-range bursts and IPD changes
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),

        # Temporal pattern — catches slower drift over the window
        LSTM(32),
        Dropout(0.3),

        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training(history, output_path='lora_training.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('CNN-LSTM Training — LoRa Attack Detection', fontweight='bold')
    for ax, key, title in zip(axes,
                               ['loss', 'auc', 'recall'],
                               ['Loss', 'AUC', 'Recall']):
        ax.plot(history.history[key],          label='Train', color='#0D9E6E')
        ax.plot(history.history[f'val_{key}'], label='Val',   color='#C0392B')
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Training plot saved to {output_path}")
    plt.show()


def plot_confusion(y_true, y_pred, threshold, output_path='lora_confusion.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='viridis')
    plt.colorbar(im, ax=ax)
    labels = ['Normal', 'Attack']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'CNN-LSTM Confusion Matrix — LoRa (threshold={threshold})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrix saved to {output_path}")
    plt.show()


def plot_threshold_sweep(y_true, y_pred_prob, output_path='lora_threshold.png'):
    """
    Shows precision, recall and F1 across all thresholds.
    The best threshold is where F1 peaks — use that value in LSTMIDS_PI.py.
    """
    thresholds = np.arange(0.1, 0.91, 0.05)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_p = (y_pred_prob >= t).astype(int)
        tp  = int(((y_true == 1) & (y_p == 1)).sum())
        fp  = int(((y_true == 0) & (y_p == 1)).sum())
        fn  = int(((y_true == 1) & (y_p == 0)).sum())
        p   = tp / (tp + fp + 1e-9)
        r   = tp / (tp + fn + 1e-9)
        f1  = 2 * p * r / (p + r + 1e-9)
        precisions.append(p); recalls.append(r); f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_t   = float(thresholds[best_idx])
    print(f"\n  Best threshold by F1: {best_t:.2f}  "
          f"(P={precisions[best_idx]:.3f}  R={recalls[best_idx]:.3f}  "
          f"F1={f1s[best_idx]:.3f})")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, precisions, label='Precision', color='#3498DB')
    ax.plot(thresholds, recalls,    label='Recall',    color='#E74C3C')
    ax.plot(thresholds, f1s,        label='F1',        color='#2ECC71')
    ax.axvline(x=best_t, color='orange', linestyle='--',
               label=f'Best F1 threshold ({best_t:.2f})')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
    ax.set_title('Threshold Sweep — Precision / Recall / F1')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Threshold sweep saved to {output_path}")
    plt.show()
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on LoRa data")
    parser.add_argument('--normal',     default='lora_normal_data.jsonl')
    parser.add_argument('--attack',     default='lora_attack_data.jsonl')
    parser.add_argument('--model-out',  default=MODEL_OUT)
    parser.add_argument('--scaler-out', default=SCALER_OUT)
    parser.add_argument('--time-steps', type=int,   default=TIME_STEPS)
    parser.add_argument('--epochs',     type=int,   default=EPOCHS)
    parser.add_argument('--threshold',  type=float, default=THRESHOLD)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CNN-LSTM TRAINER — LoRa Attack Detection (v3)")
    print("="*60)
    print(f"  Window size:  {args.time_steps} packets")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Features:     {FEATURES}")

    # ── Load data ──
    print(f"\nLoading normal data: {args.normal}")
    normal_df = load_jsonl(args.normal, expected_label=0)

    print(f"Loading attack data: {args.attack}")
    attack_df = load_jsonl(args.attack, expected_label=1)

    if len(normal_df) == 0 or len(attack_df) == 0:
        print("ERROR: Need both normal and attack records to train.")
        return

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.sort_values('arrival_local').reset_index(drop=True)
    df['ipd'] = df['arrival_local'].diff().fillna(0.0)
    df = engineer_features(df)

    print(f"\n  Total records: {len(df):,}")
    print(f"  Normal:        {(df['label']==0).sum():,}")
    print(f"  Attack:        {(df['label']==1).sum():,}")

    # ── Scale ──
    print("\nFitting RobustScaler...")
    X_raw = df[FEATURES].values
    y_raw = df['label'].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, args.scaler_out)
    print(f"  Scaler saved to {args.scaler_out}")

    # ── Sequences ──
    print(f"\nCreating sequences (window={args.time_steps})...")
    X_seq, y_seq = create_sequences(X_scaled, y_raw, args.time_steps)
    print(f"  Sequences: {X_seq.shape}  Labels: {y_seq.shape}")

    split   = int(len(X_seq) * TRAIN_SPLIT)
    X_train = X_seq[:split];  X_test = X_seq[split:]
    y_train = y_seq[:split];  y_test = y_seq[split:]
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"  Train — Normal: {int((y_train==0).sum()):,}  Attack: {int((y_train==1).sum()):,}")
    print(f"  Test  — Normal: {int((y_test==0).sum()):,}   Attack: {int((y_test==1).sum()):,}")

    # ── Class weights — standard balanced only, no extra boost ──
    weights       = compute_class_weight('balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"\n  Class weights: {class_weights}")

    # ── Build and train ──
    print("\nBuilding CNN-LSTM model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    callbacks = [
        # Monitor val_loss — stable and won't cause collapse
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(args.model_out, save_best_only=True,
                        monitor='val_auc', mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=1, min_lr=1e-6),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──
    print("\n" + "="*60)
    print(f"  EVALUATION (threshold={args.threshold})")
    print("="*60)
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(f"  {name.capitalize():12s}: {value:.4f}")

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob >= args.threshold).astype(int)
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Attack'])}")

    tn = int(((y_test==0) & (y_pred==0)).sum())
    fp = int(((y_test==0) & (y_pred==1)).sum())
    fn = int(((y_test==1) & (y_pred==0)).sum())
    tp = int(((y_test==1) & (y_pred==1)).sum())
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    print(f"  Attack recall:  {tp/(tp+fn+1e-9):.1%}")
    print(f"  False neg rate: {fn/(tp+fn+1e-9):.1%}")

    # ── Save ──
    model.save(args.model_out)
    print(f"\n  Model saved to {args.model_out}")
    print(f"  Scaler saved to {args.scaler_out}")

    # ── Charts — threshold sweep will print the best threshold ──
    plot_training(history)
    best_t = plot_threshold_sweep(y_test, y_pred_prob)
    plot_confusion(y_test, y_pred, args.threshold)

    print(f"\n  Update LORA_THRESHOLD in LSTMIDS_PI.py to {best_t:.2f}")
    print(f"  Copy {args.model_out} and {args.scaler_out} to the Pi")


if __name__ == '__main__':
    main()