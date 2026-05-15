#!/usr/bin/env python3
"""
train_model_windows.py — Train the IDS model on Windows using pandas DataFrames
Loads normal traffic from sensor_data.json, trains the anomaly detector,
validates it against attack files, saves the model as a .pkl file ready to
copy to the Pi, and produces charts of the results.

The attack data is split 80/20 — 80% is used for validation after training,
and 20% is held back as unseen data to test how the model handles new attacks
it has never encountered before.

Setup (run once in Command Prompt):
    pip install scikit-learn numpy pandas matplotlib

Run:
    python train_model_windows.py
    python train_model_windows.py --normal sensor_data.json
"""

import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Endpoint encoding — must match IDS_PI.py
ENDPOINT_MAP = {
    '/sensor':      0,
    '/status':      1,
    '/diagnostics': 2,
    '/alert':       3,
    '/join':        4,
    '/ntp':         5,
    '/':            6,
}
UNKNOWN_ENDPOINT = 99

# All attack files — add new files here as you generate them
DEFAULT_ATTACK_FILES = [
    "attack_data.json",
    "DOS_attack_data.json",
    "timesync_attack_data.json",
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_json_to_df(filepath):
    """Load a newline-delimited JSON file into a pandas DataFrame.
    Early-format records missing network fields are dropped automatically."""
    df      = pd.read_json(filepath, lines=True)
    before  = len(df)
    df      = df.dropna(subset=['timestamp', 'src_ip', 'src_port', 'dst_port'])
    ignored = before - len(df)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Loaded {len(df):,} records ({ignored} ignored) from {filepath}")
    return df


def build_features(df):
    """Derive 5 feature columns: packet_size, packet_rate, byte_rate,
    src_port, endpoint_code."""
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


def get_feature_matrix(df):
    """Return the 5-column numpy array used by the model."""
    return df[['packet_size', 'packet_rate', 'byte_rate',
               'src_port', 'endpoint_code']].to_numpy()


# ── Dataset Summary ───────────────────────────────────────────────────────────

def print_dataset_summary(df, name):
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    print(f"  Rows:          {len(df):,}")
    print(f"  Unique IPs:    {df['src_ip'].nunique()}")
    print(f"  Unique flows:  {df.groupby(['src_ip', 'dst_port']).ngroups}")
    print(f"  Time range:    {df['timestamp'].min()}  ->  {df['timestamp'].max()}")

    if 'endpoint' in df.columns:
        print(f"\n  Endpoints:")
        for ep, count in df['endpoint'].value_counts().items():
            print(f"    {ep}: {count:,}")

    if 'label' in df.columns:
        print(f"\n  Labels:")
        for label, count in df['label'].value_counts().items():
            print(f"    {label}: {count:,}")

    print(f"\n  Feature statistics:")
    cols = [c for c in ['packet_size', 'packet_rate', 'byte_rate',
                         'src_port', 'endpoint_code'] if c in df.columns]
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
    if diversity < 0.3:
        print("  WARNING: Low diversity — consider collecting more varied traffic")

    model = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
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
    """Run the model against an attack DataFrame.
    Returns a dict of {label: (detected, total)} for charting."""
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
        status = "✓" if rate >= 50 else "⚠️ "
        print(f"  {status} {label:30s}: {detected:>5}/{total:<6} detected ({rate:.1f}%)")
        total_detected += detected
        total_records  += total

    if total_records > 0:
        overall = total_detected / total_records * 100
        print(f"\n  Overall detection rate: {total_detected:,}/{total_records:,} ({overall:.1f}%)")

    print(f"{'='*60}")
    return results


# ── Charting ──────────────────────────────────────────────────────────────────

def _label_bars(ax, bars, fmt='{:.0f}%'):
    """Annotate each bar with its value."""
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, fmt.format(h),
                ha='center', va='bottom', fontsize=7)


def _clean_spines(ax):
    """Remove top and right spines for a cleaner look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_results(seen_results, unseen_results, normal_df, attack_df, output_path="ids_results.png"):
    """Generate a 4-panel results chart and save to file."""

    labels       = sorted(set(list(seen_results.keys()) + list(unseen_results.keys())))
    seen_rates   = [seen_results[l][0]   / seen_results[l][1]   * 100
                    if l in seen_results   else 0 for l in labels]
    unseen_rates = [unseen_results[l][0] / unseen_results[l][1] * 100
                    if l in unseen_results else 0 for l in labels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("IDS Model Results", fontsize=16, fontweight='bold')

    colours = {
        'detected': '#2ecc71',
        'missed':   '#e74c3c',
        'seen':     '#3498db',
        'unseen':   '#9b59b6',
        'normal':   '#7fb3d3',
        'attack':   '#e67e22',
    }

    # ── Chart 1: Detection rate by attack type (seen vs unseen) ──
    ax1   = axes[0, 0]
    x     = np.arange(len(labels))
    w     = 0.35
    bars1 = ax1.bar(x - w/2, seen_rates,   w, label='Seen (80%)',   color=colours['seen'],   alpha=0.85)
    bars2 = ax1.bar(x + w/2, unseen_rates, w, label='Unseen (20%)', color=colours['unseen'], alpha=0.85)

    ax1.set_title('Detection Rate by Attack Type', fontweight='bold')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=8)
    ax1.set_ylim(0, 115)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    _clean_spines(ax1)
    _label_bars(ax1, bars1)
    _label_bars(ax1, bars2)

    # ── Chart 2: Detected vs missed — overall pie ──
    ax2 = axes[0, 1]
    total_detected = sum(v[0] for v in seen_results.values()) + \
                     sum(v[0] for v in unseen_results.values())
    total_records  = sum(v[1] for v in seen_results.values()) + \
                     sum(v[1] for v in unseen_results.values())
    total_missed   = total_records - total_detected

    wedges, texts, autotexts = ax2.pie(
        [total_detected, total_missed],
        labels=['Detected', 'Missed'],
        colors=[colours['detected'], colours['missed']],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax2.set_title(f'Overall Detection\n({total_detected:,} / {total_records:,} records)',
                  fontweight='bold')

    # ── Chart 3: Attack type distribution in training data ──
    ax3 = axes[1, 0]
    if 'label' in attack_df.columns:
        label_counts  = attack_df['label'].value_counts()
        bar_colours   = plt.cm.Set2(np.linspace(0, 1, len(label_counts)))
        tick_labels   = [l.replace('_', ' ') for l in label_counts.index]
        bars          = ax3.barh(tick_labels, label_counts.values,
                                 color=bar_colours, alpha=0.85)
        offset        = label_counts.values.max() * 0.01
        ax3.set_title('Attack Records by Type', fontweight='bold')
        ax3.set_xlabel('Number of Records')
        ax3.grid(axis='x', alpha=0.3)
        _clean_spines(ax3)
        for bar, val in zip(bars, label_counts.values):
            ax3.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                     f'{val:,}', va='center', fontsize=8)

    # ── Chart 4: Feature distributions — normal vs attack ──
    ax4 = axes[1, 1]
    normal_rates = normal_df['packet_rate'].clip(upper=1100)
    attack_rates = attack_df['packet_rate'].clip(upper=1100)

    ax4.hist(normal_rates, bins=50, alpha=0.6, color=colours['normal'],
             label=f'Normal ({len(normal_df):,} records)', density=True)
    ax4.hist(attack_rates, bins=50, alpha=0.6, color=colours['attack'],
             label=f'Attack ({len(attack_df):,} records)', density=True)
    ax4.set_title('Packet Rate Distribution\nNormal vs Attack', fontweight='bold')
    ax4.set_xlabel('Packet Rate (packets/sec)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    _clean_spines(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved to {output_path}")
    plt.show()


# ── Save Model ────────────────────────────────────────────────────────────────

def save_model(model, filepath):
    model_data = {
        'anomaly_detector': model,
        'timestamp':        datetime.now().isoformat(),
        'training_source':  'sensor_data.json (Windows offline training)',
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to {filepath}")
    print(f"  Copy this file to your Pi alongside IDS_PI.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IDS model on Windows from JSON data")
    parser.add_argument("--normal",     default="sensor_data.json")
    parser.add_argument("--attacks",    nargs="+", default=DEFAULT_ATTACK_FILES)
    parser.add_argument("--output",     default="ids_model.pkl")
    parser.add_argument("--chart",      default="ids_results.png")
    parser.add_argument("--test-split", type=float, default=0.2)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("IDS MODEL TRAINER — Windows Edition")
    print("="*60)

    # ── Load normal traffic ──
    print(f"\nLoading normal traffic: {args.normal}...")
    normal_df = load_json_to_df(args.normal)
    normal_df = build_features(normal_df)
    print_dataset_summary(normal_df, "Normal Traffic")

    if len(normal_df) < 100:
        print(f"  Not enough normal records (need 100+, got {len(normal_df)})")
        return

    # ── Load all attack files ──
    attack_frames = []
    for filepath in args.attacks:
        try:
            print(f"\nLoading attack data: {filepath}...")
            df = load_json_to_df(filepath)
            df = build_features(df)
            attack_frames.append(df)
        except FileNotFoundError:
            print(f"  (File not found: {filepath} — skipping)")
        except Exception as e:
            print(f"  (Error loading {filepath}: {e} — skipping)")

    if not attack_frames:
        print("\n  No attack files loaded — skipping validation")
        model = train(normal_df)
        save_model(model, args.output)
        return

    attack_df = pd.concat(attack_frames, ignore_index=True)
    print_dataset_summary(attack_df, "Attack Data (combined)")

    # ── Split attack data 80/20 into seen vs unseen ──
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
    print(f"  Unseen (test):      {len(unseen_df):,} records  (20% — held back)")
    if 'label' in attack_df.columns:
        print(f"\n  Unseen records by label:")
        for label, count in unseen_df['label'].value_counts().items():
            print(f"    {label}: {count:,}")
    print(f"{'='*60}")

    # ── Train ──
    model = train(normal_df)

    # ── Validate ──
    seen_results   = validate(model, seen_df,
                               title="VALIDATION — SEEN ATTACK DATA (80%)")
    unseen_results = validate(model, unseen_df,
                               title="TEST — UNSEEN ATTACK DATA (20%)")

    # ── Chart ──
    plot_results(seen_results, unseen_results, normal_df, attack_df,
                 output_path=args.chart)

    # ── Save ──
    save_model(model, args.output)


if __name__ == "__main__":
    main()