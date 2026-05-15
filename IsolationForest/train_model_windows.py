#!/usr/bin/env python3
"""
train_model_windows.py — Train the IDS model on Windows using pandas DataFrames
Loads normal traffic from sensor_data.json, trains the anomaly detector,
validates it against attack files, saves the model as a .pkl file ready to
copy to the Pi, and produces charts of the results.

Now includes LoRa radio traffic features (ts, dt, rssi, snr) merged into
the same IsolationForest model alongside HTTP traffic features.

The attack data is split 80/20 — 80% is used for validation after training,
and 20% is held back as unseen data to test how the model handles new attacks
it has never encountered before.

Setup (run once in Command Prompt):
    pip install scikit-learn numpy pandas matplotlib

Run:
    python train_model_windows.py
    python train_model_windows.py --normal sensor_data.json
    python train_model_windows.py --lora-normal lora_normal_data.json --lora-attack lora_attack_data.json
"""

import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Endpoint encoding — must match IDS_IsolationForest_PI4.py
ENDPOINT_MAP = {
    '/sensor':      0,
    '/status':      1,
    '/diagnostics': 2,
    '/alert':       3,
    '/join':        4,
    '/ntp':         5,
    '/lora':        7,
    '/':            6,
}
UNKNOWN_ENDPOINT = 99

# HTTP attack files
DEFAULT_ATTACK_FILES = [
    "attack_data.json",
    "DOS_attack_data.json",
    "attack_port_scan.json",
    "attack_slow_loris.json",
    "attack_lora_exploit.json",
    "attack_time_sync.json",
]

# LoRa data files
DEFAULT_LORA_NORMAL = "lora_normal_data.json"
DEFAULT_LORA_ATTACK = "lora_attack_data.json"

# Feature columns used by the model
# HTTP features:    packet_size, packet_rate, byte_rate, src_port, endpoint_code
# LoRa features:    packet_size(rssi-mapped), packet_rate(dt-mapped),
#                   byte_rate(snr-mapped), src_port(0), endpoint_code(7=/lora)
# Same 5 columns — LoRa values are mapped to the same feature space
FEATURE_COLS = ['packet_size', 'packet_rate', 'byte_rate', 'src_port', 'endpoint_code']


# ── HTTP Data Loading ─────────────────────────────────────────────────────────

def load_json_to_df(filepath):
    """Load a newline-delimited JSON file into a pandas DataFrame."""
    df     = pd.read_json(filepath, lines=True)
    before = len(df)
    df     = df.dropna(subset=['timestamp', 'src_ip', 'src_port', 'dst_port'])
    ignored = before - len(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Loaded {len(df):,} records ({ignored} ignored) from {filepath}")
    return df


def build_http_features(df):
    """Derive 5 HTTP feature columns."""
    meta_cols         = {'timestamp', 'src_ip', 'src_port', 'dst_port',
                         'endpoint', 'label', 'pico_id'}
    payload_col_count = len([c for c in df.columns if c not in meta_cols])
    df = df.copy()

    if 'packet_size' in df.columns:
        df['packet_size'] = df['packet_size'].fillna(
            40 + payload_col_count * 20).clip(lower=64)
    else:
        df['packet_size'] = max(64, 40 + payload_col_count * 20)

    ts_epoch       = df['timestamp'].astype('int64') / 1e9
    df['interval'] = ts_epoch.diff().fillna(1.0).clip(lower=0.001)

    df['packet_rate'] = 1.0 / df['interval']
    df['byte_rate']   = df['packet_size'] / df['interval']
    df['src_port']    = df['src_port'].fillna(0).clip(lower=0, upper=65535)

    if 'endpoint' in df.columns:
        df['endpoint_code'] = df['endpoint'].map(
            ENDPOINT_MAP).fillna(UNKNOWN_ENDPOINT).astype(int)
    else:
        df['endpoint_code'] = 0

    return df


# ── LoRa Data Loading ─────────────────────────────────────────────────────────

def load_lora_to_df(filepath):
    """Load a LoRa JSON file (ts, dt, rssi, snr, label) into a DataFrame."""
    df = pd.read_json(filepath, lines=True)
    before = len(df)
    df = df.dropna(subset=['ts', 'rssi', 'snr'])
    ignored = before - len(df)
    print(f"  Loaded {len(df):,} LoRa records ({ignored} ignored) from {filepath}")
    return df


def build_lora_features(df):
    """
    Map LoRa radio features into the same 5-column feature space as HTTP:

      packet_size   <- RSSI remapped to positive range (stronger signal = larger)
                       rssi is negative dBm (e.g. -80), so we use abs(rssi)
      packet_rate   <- inverse of dt (inter-packet interval in seconds)
                       dt=0 or very small → high rate (attack-like)
      byte_rate     <- SNR as a proxy for signal quality
      src_port      <- 0 (LoRa has no TCP src port)
      endpoint_code <- 7 (/lora endpoint)
    """
    df = df.copy()

    # packet_size: abs(rssi) — ranges ~20 to 120 (weak to strong)
    df['packet_size'] = df['rssi'].abs().clip(lower=20, upper=200)

    # packet_rate: 1 / dt — clamp dt to at least 0.1s to avoid div/0
    # Normal dt ~10s → rate ~0.1; Attack dt ~0.5s → rate ~2.0
    df['packet_rate'] = 1.0 / df['dt'].clip(lower=0.1)

    # byte_rate: snr value (higher SNR = cleaner signal)
    df['byte_rate'] = df['snr'].clip(lower=-20, upper=40)

    df['src_port']      = 0
    df['endpoint_code'] = ENDPOINT_MAP['/lora']

    # Ensure label column is consistent
    if 'label' in df.columns:
        df['label'] = df['label'].apply(
            lambda x: 'lora_attack' if x == 1 or x == 'lora_attack' else 'lora_normal')
    else:
        df['label'] = 'lora_normal'

    return df


def get_feature_matrix(df):
    """Return the 5-column numpy array used by the model."""
    return df[FEATURE_COLS].to_numpy()


# ── Dataset Summary ───────────────────────────────────────────────────────────

def print_dataset_summary(df, name):
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    print(f"  Rows:    {len(df):,}")
    if 'label' in df.columns:
        print(f"\n  Labels:")
        for label, count in df['label'].value_counts().items():
            print(f"    {label}: {count:,}")
    print(f"\n  Feature statistics:")
    cols = [c for c in FEATURE_COLS if c in df.columns]
    print(df[cols].describe().to_string())
    print(f"{'='*60}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(normal_df):
    """Train IsolationForest on the normal traffic DataFrame."""
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    features = get_feature_matrix(normal_df)
    n        = len(features)
    print(f"  Samples:  {n:,}")

    unique    = len(np.unique(features, axis=0))
    diversity = unique / n
    print(f"  Unique samples: {unique:,} ({diversity*100:.1f}% diversity)")

    model = IsolationForest(contamination=0.2, random_state=42, n_estimators=100)
    model.fit(features)

    if n >= 100000:   quality = "EXCELLENT"
    elif n >= 50000:  quality = "GOOD"
    elif n >= 10000:  quality = "ACCEPTABLE"
    elif n >= 5000:   quality = "MARGINAL"
    else:             quality = "POOR"

    print(f"\n  Training quality: {quality}")
    print(f"{'='*60}")
    return model


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, attack_df, title="VALIDATION"):
    """Run the model against an attack DataFrame."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    label_col = attack_df['label'] if 'label' in attack_df.columns else pd.Series(
        ['unknown'] * len(attack_df), index=attack_df.index)

    results        = {}
    total_detected = 0
    total_records  = 0

    for label in sorted(label_col.unique()):
        subset      = attack_df[label_col == label]
        features    = get_feature_matrix(subset)
        if len(features) == 0:
            continue
        predictions = model.predict(features)
        detected    = int(np.sum(predictions == -1))
        total       = len(predictions)
        rate        = detected / total * 100
        results[label] = (detected, total)
        status = "✓" if rate >= 50 else "⚠ "
        print(f"  {status} {label:35s}: {detected:>5}/{total:<6} ({rate:.1f}%)")
        total_detected += detected
        total_records  += total

    if total_records > 0:
        overall = total_detected / total_records * 100
        print(f"\n  Overall: {total_detected:,}/{total_records:,} ({overall:.1f}%)")

    print(f"{'='*60}")
    return results


# ── Charting ──────────────────────────────────────────────────────────────────

def plot_results(seen_results, unseen_results, normal_df, attack_df,
                 lora_normal_df=None, lora_attack_df=None,
                 output_path="ids_results.png"):
    """Generate results charts and save to file."""

    exclude      = {'fake_sensor'}
    seen_results   = {k: v for k, v in seen_results.items()   if k not in exclude}
    unseen_results = {k: v for k, v in unseen_results.items() if k not in exclude}
    if 'label' in attack_df.columns:
        attack_df = attack_df[~attack_df['label'].isin(exclude)]

    labels       = sorted(set(list(seen_results.keys()) + list(unseen_results.keys())))
    seen_rates   = [seen_results[l][0]   / seen_results[l][1]   * 100
                    if l in seen_results   else 0 for l in labels]
    unseen_rates = [unseen_results[l][0] / unseen_results[l][1] * 100
                    if l in unseen_results else 0 for l in labels]

    colours = {
        'detected': '#2ecc71', 'missed':  '#e74c3c',
        'seen':     '#3498db', 'unseen':  '#9b59b6',
        'normal':   '#1abc9c', 'attack':  '#e67e22',
        'lora_n':   '#27ae60', 'lora_a':  '#c0392b',
    }

    n_charts = 5 if lora_normal_df is not None else 4
    fig, axes = plt.subplots(2, 3 if n_charts == 5 else 2,
                             figsize=(18 if n_charts == 5 else 14, 10))
    axes = axes.flatten()
    fig.suptitle("IDS Model Results", fontsize=16, fontweight='bold')

    # Chart 1: Detection rate by attack type
    ax1 = axes[0]
    x   = np.arange(len(labels))
    w   = 0.35
    b1  = ax1.bar(x - w/2, seen_rates,   w, label='Seen (80%)',   color=colours['seen'],   alpha=0.85)
    b2  = ax1.bar(x + w/2, unseen_rates, w, label='Unseen (20%)', color=colours['unseen'], alpha=0.85)
    ax1.set_title('Detection Rate by Attack Type', fontweight='bold')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=7)
    ax1.set_ylim(0, 115)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%',
                 ha='center', va='bottom', fontsize=6)

    # Chart 2: Overall detection pie
    ax2 = axes[1]
    total_det = sum(v[0] for v in list(seen_results.values()) + list(unseen_results.values()))
    total_rec = sum(v[1] for v in list(seen_results.values()) + list(unseen_results.values()))
    total_mis = total_rec - total_det
    wedges, texts, autotexts = ax2.pie(
        [total_det, total_mis],
        labels=['Detected', 'Missed'],
        colors=[colours['detected'], colours['missed']],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight('bold')
    ax2.set_title(f'Overall Detection\n({total_det:,} / {total_rec:,} records)',
                  fontweight='bold')

    # Chart 3: Attack type distribution
    ax3 = axes[2]
    if 'label' in attack_df.columns:
        lc   = attack_df['label'].value_counts()
        bcs  = plt.cm.Set2(np.linspace(0, 1, len(lc)))
        bars = ax3.barh(lc.index, lc.values, color=bcs, alpha=0.85)
        ax3.set_title('Attack Records by Type', fontweight='bold')
        ax3.set_xlabel('Number of Records')
        ax3.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, lc.values):
            ax3.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                     f'{val:,}', va='center', fontsize=8)

    # Chart 4: Packet rate distribution
    ax4 = axes[3]
    ax4.hist(normal_df['packet_rate'].clip(upper=1100), bins=50,
             alpha=0.6, color=colours['normal'],
             label=f'HTTP normal ({len(normal_df):,})', density=True)
    ax4.hist(attack_df['packet_rate'].clip(upper=1100), bins=50,
             alpha=0.6, color=colours['attack'],
             label=f'HTTP attack ({len(attack_df):,})', density=True)
    if lora_normal_df is not None:
        ax4.hist(lora_normal_df['packet_rate'].clip(upper=1100), bins=30,
                 alpha=0.5, color=colours['lora_n'],
                 label=f'LoRa normal ({len(lora_normal_df):,})', density=True)
    if lora_attack_df is not None:
        ax4.hist(lora_attack_df['packet_rate'].clip(upper=1100), bins=30,
                 alpha=0.5, color=colours['lora_a'],
                 label=f'LoRa attack ({len(lora_attack_df):,})', density=True)
    ax4.set_title('Packet Rate Distribution', fontweight='bold')
    ax4.set_xlabel('Packet Rate (packets/sec)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # Chart 5: LoRa DT distribution (if available)
    if lora_normal_df is not None and n_charts == 5:
        ax5 = axes[4]
        ax5.hist(lora_normal_df['dt'].clip(upper=30), bins=40,
                 alpha=0.7, color=colours['lora_n'],
                 label=f'Normal ({len(lora_normal_df):,})')
        if lora_attack_df is not None:
            ax5.hist(lora_attack_df['dt'].clip(upper=30), bins=40,
                     alpha=0.7, color=colours['lora_a'],
                     label=f'Attack ({len(lora_attack_df):,})')
        ax5.set_title('LoRa Inter-Packet Interval (DT)', fontweight='bold')
        ax5.set_xlabel('DT (seconds)')
        ax5.set_ylabel('Count')
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.3)

    # Hide any unused axes
    for i in range(n_charts, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved to {output_path}")
    plt.show()


# ── Save Model ────────────────────────────────────────────────────────────────

def save_model(model, filepath, meta=None):
    model_data = {
        'anomaly_detector': model,
        'timestamp':        datetime.now().isoformat(),
        'training_source':  'sensor_data.json + lora_normal_data.json',
        'feature_cols':     FEATURE_COLS,
    }
    if meta:
        model_data.update(meta)
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to {filepath}")
    print(f"  Copy this file to your Pi as ids_model.pkl alongside IDS_IsolationForest_PI4.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IDS model from HTTP + LoRa data")
    parser.add_argument("--normal",       default="sensor_data.json")
    parser.add_argument("--attacks",      nargs="+", default=DEFAULT_ATTACK_FILES)
    parser.add_argument("--lora-normal",  default=DEFAULT_LORA_NORMAL)
    parser.add_argument("--lora-attack",  default=DEFAULT_LORA_ATTACK)
    parser.add_argument("--output",       default="ids_model.pkl")
    parser.add_argument("--chart",        default="ids_results.png")
    parser.add_argument("--test-split",   type=float, default=0.2)
    parser.add_argument("--no-lora",      action="store_true",
                        help="Skip LoRa data even if files exist")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("IDS MODEL TRAINER — Windows Edition  (HTTP + LoRa)")
    print("="*60)

    # ── Load HTTP normal traffic ──
    print(f"\nLoading HTTP normal traffic: {args.normal}...")
    normal_df = load_json_to_df(args.normal)
    normal_df = build_http_features(normal_df)

    # ── Load LoRa normal traffic ──
    lora_normal_df = None
    if not args.no_lora:
        try:
            print(f"\nLoading LoRa normal traffic: {args.lora_normal}...")
            lora_normal_df = load_lora_to_df(args.lora_normal)
            lora_normal_df = build_lora_features(lora_normal_df)
            print(f"  LoRa normal: {len(lora_normal_df):,} records")
        except FileNotFoundError:
            print(f"  (File not found: {args.lora_normal} — skipping LoRa normal data)")
        except Exception as e:
            print(f"  (Error loading LoRa normal data: {e} — skipping)")

    # ── Merge normal data ──
    if lora_normal_df is not None:
        all_normal_df = pd.concat([normal_df, lora_normal_df], ignore_index=True)
        print(f"\n  Combined normal: {len(normal_df):,} HTTP + {len(lora_normal_df):,} LoRa"
              f" = {len(all_normal_df):,} total")
    else:
        all_normal_df = normal_df

    print_dataset_summary(all_normal_df, "Normal Traffic (combined)")

    if len(all_normal_df) < 100:
        print(f"  Not enough normal records (need 100+, got {len(all_normal_df)})")
        return

    # ── Load HTTP attack files ──
    attack_frames = []
    for filepath in args.attacks:
        try:
            print(f"\nLoading HTTP attack data: {filepath}...")
            df = load_json_to_df(filepath)
            df = build_http_features(df)
            attack_frames.append(df)
        except FileNotFoundError:
            print(f"  (File not found: {filepath} — skipping)")
        except Exception as e:
            print(f"  (Error loading {filepath}: {e} — skipping)")

    # ── Load LoRa attack data ──
    lora_attack_df = None
    if not args.no_lora:
        try:
            print(f"\nLoading LoRa attack data: {args.lora_attack}...")
            lora_attack_df = load_lora_to_df(args.lora_attack)
            lora_attack_df = build_lora_features(lora_attack_df)
            print(f"  LoRa attack: {len(lora_attack_df):,} records")
            attack_frames.append(lora_attack_df)
        except FileNotFoundError:
            print(f"  (File not found: {args.lora_attack} — skipping LoRa attack data)")
        except Exception as e:
            print(f"  (Error loading LoRa attack data: {e} — skipping)")

    if not attack_frames:
        print("\n  No attack files loaded — training on normal data only")
        model = train(all_normal_df)
        save_model(model, args.output)
        return

    attack_df = pd.concat(attack_frames, ignore_index=True)
    print_dataset_summary(attack_df, "Attack Data (combined)")

    # ── Split attack data 80/20 ──
    seen_df, unseen_df = train_test_split(
        attack_df,
        test_size=args.test_split,
        random_state=42,
        stratify=attack_df['label'] if 'label' in attack_df.columns else None
    )

    print(f"\n{'='*60}")
    print("ATTACK DATA SPLIT")
    print(f"{'='*60}")
    print(f"  Seen (validation):  {len(seen_df):,} records  (80%)")
    print(f"  Unseen (test):      {len(unseen_df):,} records  (20%)")
    if 'label' in attack_df.columns:
        print(f"\n  Unseen records by label:")
        for label, count in unseen_df['label'].value_counts().items():
            print(f"    {label}: {count:,}")
    print(f"{'='*60}")

    # ── Train ──
    model = train(all_normal_df)

    # ── Validate ──
    seen_results   = validate(model, seen_df,
                               title="VALIDATION — SEEN ATTACK DATA (80%)")
    unseen_results = validate(model, unseen_df,
                               title="TEST — UNSEEN ATTACK DATA (20%)")

    # ── Chart ──
    plot_results(seen_results, unseen_results, normal_df, attack_df,
                 lora_normal_df=lora_normal_df, lora_attack_df=lora_attack_df,
                 output_path=args.chart)

    # ── Save ──
    save_model(model, args.output, meta={
        'http_normal_count': len(normal_df),
        'lora_normal_count': len(lora_normal_df) if lora_normal_df is not None else 0,
        'attack_count':      len(attack_df),
    })


if __name__ == "__main__":
    main()
