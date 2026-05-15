#!/usr/bin/env python3
"""
lora_server.py — Standalone LoRa data collection server for Raspberry Pi

Receives JSON from the receiver Pico (r_pico_lora.py) via HTTP POST
and saves records to two JSONL files based on the label field:

    lora_normal_data.jsonl  — label=0 (normal traffic)
    lora_attack_data.jsonl  — label=1 (attack traffic)

These files are copied to a Windows/desktop machine and used by
train_lora_lstm.py to train the CNN-LSTM model. No training happens here.

Run:
    sudo /home/t-rex/Documents/ids_env/bin/python3 /home/t-rex/Documents/lora_server.py

Note: uses port 80 — stop IDS_IsolationForest_PI4.py before running this.
"""

import json
import argparse
import logging
from datetime import datetime
from flask import Flask, request, jsonify

# ── Config ────────────────────────────────────────────────────────────────────
HOST         = '0.0.0.0'
PORT         = 80
NORMAL_FILE  = 'lora_normal_data.jsonl'
ATTACK_FILE  = 'lora_attack_data.jsonl'

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('LoRa')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

counts = {'normal': 0, 'attack': 0}
app    = Flask(__name__)


@app.route('/lora', methods=['POST'])
def lora():
    data  = request.get_json(silent=True, force=True) or {}
    label = int(data.get('label', 0))
    ts    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Extract fields from the rich Pico payload ─────────────────────────────
    metadata    = data.get('metadata',      {})
    timestamps  = data.get('timestamps',    {})
    phys        = data.get('physical_layer',{})

    entry = {
        "metadata": {
            "seq":     metadata.get('seq',     0),
            "node_id": metadata.get('node_id', 'unknown'),
            "saved_at": ts,
        },
        "timestamps": {
            "transmitted_real": timestamps.get('transmitted_real', 0),
            "transmitted_sent": timestamps.get('transmitted_sent', 0),
            "arrival_local":    timestamps.get('arrival_local',    0),
            "delta_t":          timestamps.get('delta_t',          0),
        },
        "physical_layer": {
            "hw_rssi":  phys.get('hw_rssi',  0),
            "hw_snr":   phys.get('hw_snr',   0),
            "sim_rssi": phys.get('sim_rssi', 0),
            "sim_snr":  phys.get('sim_snr',  0),
        },
        "ipd":   data.get('ipd',   0),
        "label": label,
    }

    if label == 1:
        outfile = ATTACK_FILE
        counts['attack'] += 1
        log.warning(
            f"ATTACK  SEQ={entry['metadata']['seq']:>5}  "
            f"RSSI={entry['physical_layer']['hw_rssi']:>5}  "
            f"SNR={entry['physical_layer']['hw_snr']:>5}  "
            f"DT={entry['timestamps']['delta_t']:>8.3f}  "
            f"IPD={entry['ipd']:>6.2f}  total={counts['attack']}"
        )
    else:
        outfile = NORMAL_FILE
        counts['normal'] += 1
        log.info(
            f"NORMAL  SEQ={entry['metadata']['seq']:>5}  "
            f"RSSI={entry['physical_layer']['hw_rssi']:>5}  "
            f"SNR={entry['physical_layer']['hw_snr']:>5}  "
            f"DT={entry['timestamps']['delta_t']:>8.3f}  "
            f"IPD={entry['ipd']:>6.2f}  total={counts['normal']}"
        )

    with open(outfile, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    return jsonify({'status': 'ok'}), 200


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status':       'running',
        'normal_saved': counts['normal'],
        'attack_saved': counts['attack'],
        'normal_file':  NORMAL_FILE,
        'attack_file':  ATTACK_FILE,
    }), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'LoRa data collection server'}), 200


def main():
    global NORMAL_FILE, ATTACK_FILE
    parser = argparse.ArgumentParser(description="LoRa data collection server")
    parser.add_argument('--host',         default=HOST)
    parser.add_argument('--port',         type=int, default=PORT)
    parser.add_argument('--normal-file',  default=NORMAL_FILE)
    parser.add_argument('--attack-file',  default=ATTACK_FILE)
    args = parser.parse_args()

    NORMAL_FILE = args.normal_file
    ATTACK_FILE = args.attack_file

    print("\n" + "="*55)
    print("  LoRa Data Collection Server")
    print("="*55)
    print(f"  Listening on:  {args.host}:{args.port}")
    print(f"  Normal file:   {NORMAL_FILE}")
    print(f"  Attack file:   {ATTACK_FILE}")
    print(f"  Endpoint:      POST /lora")
    print(f"  Check counts:  GET  /status")
    print("="*55 + "\n")

    app.run(host=args.host, port=args.port, use_reloader=False, threaded=True)


if __name__ == '__main__':
    main()