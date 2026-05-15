#!/usr/bin/env python3
"""
IDS_PI.py — HTTP IDS/IPS for Raspberry Pi
Uses stdlib HTTPServer + IsolationForest anomaly detection.
Terminal GUI with live stats, colored output, and anomaly scores.

Requirements:
    pip install scikit-learn numpy

Run:
    sudo python3 IDS_PI.py
    sudo python3 IDS_PI.py --model ids_model.pkl --port 80
"""

import pickle
import logging
import argparse
import threading
import json
import numpy as np
import warnings
import resource
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

warnings.filterwarnings('ignore', category=ResourceWarning)

# Raise file descriptor limit to handle connection floods
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
except Exception:
    pass

# ── Endpoint encoding — must match training ───────────────────────────────────
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

# ── IPS settings ──────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD       = 5
BAN_DURATION_SECS       = 120
HEADER_SIZE_BYTES       = 200
ANOMALY_SCORE_THRESHOLD = -0.5
SLOWLORIS_CONN_LIMIT    = 10   # ban IP if it holds more than this many open connections

# ── Whitelist — Pi's own IPs, never flagged ───────────────────────────────────
WHITELIST = {
    '10.42.0.1',
    '10.46.1.40',
    '127.0.0.1',
}

# ── HTTP passthrough — monitored but never blocked ────────────────────────────
HTTP_PASSTHROUGH = {
    '10.42.0.71',   # Pico-2
    '10.42.0.211',  # Pico-3
    '10.42.0.215',  # Pico-1
}

# ── Terminal colors ───────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    PURPLE = "\033[95m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"
    ORANGE = "\033[33m"

# ── Live stats ────────────────────────────────────────────────────────────────
stats = {
    "allowed":   0,
    "blocked":   0,
    "anomalies": 0,
    "banned":    0,
    "monitor":   0,
}
stats_lock = threading.Lock()

# ── Print helpers ─────────────────────────────────────────────────────────────
def _ts():
    return datetime.now().strftime('%H:%M:%S')

def print_header():
    print(f"\n{C.CYAN}{C.BOLD}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║      IsolationForest IDS — Raspberry Pi              ║")
    print("  ║      HTTP Layer 7 Traffic Monitor                    ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")

def print_separator():
    print(f"  {C.GRAY}{'─'*70}{C.RESET}")

def print_status_bar():
    with stats_lock:
        a = stats['allowed']
        b = stats['blocked']
        n = stats['anomalies']
        ban = stats['banned']
        m = stats['monitor']
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET}  "
        f"{C.GREEN}OK:{a:>5}{C.RESET}  "
        f"{C.RED}Blocked:{b:>5}{C.RESET}  "
        f"{C.YELLOW}Anomalies:{n:>4}{C.RESET}  "
        f"{C.ORANGE}Monitor:{m:>4}{C.RESET}  "
        f"{C.RED}Banned:{ban:>3}{C.RESET}"
    )

def log_ok(ip, method, path, score, extra=""):
    with stats_lock:
        stats["allowed"] += 1
    score_str = f"{C.GRAY}score={score:+.3f}{C.RESET}"
    extra_str = f"  {C.GRAY}{extra}{C.RESET}" if extra else ""
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.GREEN}[ OK      ]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.WHITE}{method:<5}{C.RESET} "
        f"{C.CYAN}{path:<15}{C.RESET}  "
        f"{score_str}"
        f"{extra_str}"
    )

def log_monitor(ip, method, path, score):
    with stats_lock:
        stats["monitor"] += 1
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.ORANGE}[ MONITOR ]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.WHITE}{method:<5}{C.RESET} "
        f"{C.CYAN}{path:<15}{C.RESET}  "
        f"{C.ORANGE}score={score:+.3f}{C.RESET}  "
        f"{C.GRAY}(passthrough){C.RESET}"
    )

def log_anomaly(ip, method, path, score, strike):
    with stats_lock:
        stats["anomalies"] += 1
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.YELLOW}[ ANOMALY ]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.WHITE}{method:<5}{C.RESET} "
        f"{C.CYAN}{path:<15}{C.RESET}  "
        f"{C.YELLOW}score={score:+.3f}{C.RESET}  "
        f"{C.YELLOW}strike {strike}/{ANOMALY_THRESHOLD}{C.RESET}"
    )

def log_banned(ip):
    with stats_lock:
        stats["banned"] += 1
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.RED}{C.BOLD}[ BANNED  ]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.RED}banned for {BAN_DURATION_SECS}s{C.RESET}"
    )

def log_blocked(ip, method, path):
    with stats_lock:
        stats["blocked"] += 1
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.RED}{C.BOLD}[ BLOCKED ]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.WHITE}{method:<5}{C.RESET} "
        f"{C.CYAN}{path:<15}{C.RESET}  "
        f"{C.RED}(banned){C.RESET}"
    )

def log_slowloris(ip, conn_count):
    with stats_lock:
        stats["anomalies"] += 1
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.YELLOW}[ SLOWLORI]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.YELLOW}{conn_count} open connections — Slow Loris suspected{C.RESET}"
    )

def log_join(ip, pico_id):
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{C.CYAN}[ JOIN    ]{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.GRAY}pico={pico_id} connected{C.RESET}"
    )

def log_lora(ip, label, rssi, snr, dt):
    color = C.RED if label == 'lora_attack' else C.GREEN
    tag   = "[ LORA ATK]" if label == 'lora_attack' else "[ LORA OK ]"
    print(
        f"  {C.GRAY}[{_ts()}]{C.RESET} "
        f"{color}{tag}{C.RESET} "
        f"{C.BLUE}{ip:<16}{C.RESET} "
        f"{C.GRAY}RSSI={rssi} SNR={snr} DT={dt}{C.RESET}"
    )

# ── File logger ───────────────────────────────────────────────────────────────
file_log = logging.getLogger('IDS_FILE')
file_log.setLevel(logging.INFO)
fh = logging.FileHandler('ids_alerts.log')
fh.setFormatter(logging.Formatter(
    '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
file_log.addHandler(fh)
file_log.propagate = False

# ── Data log ──────────────────────────────────────────────────────────────────
LOG_FILE = 'sensor_data.json'

def save_data(data):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(data) + '\n')

def build_entry(src_ip, src_port, dst_port, data, endpoint):
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['src_ip']    = src_ip
    data['src_port']  = src_port
    data['dst_port']  = dst_port
    data['endpoint']  = endpoint
    data['label']     = 'normal'
    return data

# ── Traffic state ─────────────────────────────────────────────────────────────
class TrafficState:
    def __init__(self):
        self.last_seen         = defaultdict(lambda: None)
        self.strikes           = defaultdict(int)
        self.banned_until      = defaultdict(lambda: None)
        self._lock             = threading.Lock()
        self._last_anomaly_log  = defaultdict(lambda: None)
        self._last_ban_log      = defaultdict(lambda: None)
        self._last_blocked_log  = defaultdict(lambda: None)
        self._log_interval      = 10
        self._blocked_interval  = 10  # print BLOCKED at most once per 10s per IP
        self.open_connections   = defaultdict(int)   # concurrent connections per IP
        self._conn_lock         = threading.Lock()

    def open_connection(self, ip):
        with self._conn_lock:
            self.open_connections[ip] += 1
            return self.open_connections[ip]

    def close_connection(self, ip):
        with self._conn_lock:
            self.open_connections[ip] = max(0, self.open_connections[ip] - 1)

    def is_banned(self, ip):
        if ip in WHITELIST:
            return False
        with self._lock:
            until = self.banned_until[ip]
            if until and datetime.now() < until:
                return True
            if until:
                self.banned_until[ip] = None
                self.strikes[ip]      = 0
                with stats_lock:
                    stats["banned"] = max(0, stats["banned"] - 1)
            return False

    def record_anomaly(self, ip, score, method, path):
        if ip in WHITELIST:
            return
        if self.is_banned(ip):
            return
        with self._lock:
            self.strikes[ip] += 1
            strikes  = self.strikes[ip]

        log_anomaly(ip, method, path, score, strikes)
        file_log.warning(
            f"ANOMALY {ip} {method} {path} "
            f"score={score:+.3f} strike={strikes}/{ANOMALY_THRESHOLD}"
        )

        if strikes >= ANOMALY_THRESHOLD:
            with self._lock:
                now      = datetime.now()
                last_ban = self._last_ban_log[ip]
                expired  = (last_ban is None or
                            (now - last_ban).total_seconds() >= BAN_DURATION_SECS)
                if expired:
                    self.strikes[ip]       = 0
                    self.banned_until[ip]  = now + timedelta(seconds=BAN_DURATION_SECS)
                    self._last_ban_log[ip] = now
                    do_ban = True
                else:
                    do_ban = False
            if do_ban:
                log_banned(ip)
                file_log.warning(f"BANNED {ip} for {BAN_DURATION_SECS}s")

    def get_interval(self, ip):
        now = datetime.now()
        with self._lock:
            last = self.last_seen[ip]
            self.last_seen[ip] = now
        if last is None:
            return 1.0
        return max((now - last).total_seconds(), 0.001)


state = TrafficState()

# ── Shared model ──────────────────────────────────────────────────────────────
model = None

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(handler, content_length, path):
    src_ip      = handler.client_address[0]
    src_port    = handler.client_address[1]
    packet_size = max(64, content_length + HEADER_SIZE_BYTES)
    interval    = state.get_interval(src_ip)
    packet_rate = 1.0 / interval
    byte_rate   = packet_size / interval
    endpoint_code = ENDPOINT_MAP.get(path, UNKNOWN_ENDPOINT)
    return np.array([[packet_size, packet_rate, byte_rate,
                      src_port, endpoint_code]], dtype=float)

# ── IDS check ─────────────────────────────────────────────────────────────────
def ids_check(handler, content_length, path):
    """
    Returns (blocked: bool, score: float)
    score is the raw IsolationForest decision_function value.
    Negative = anomalous, positive = normal.
    """
    src_ip = handler.client_address[0]
    method = handler.command

    if state.is_banned(src_ip):
        # Only print BLOCKED once per 10s per IP — not on every packet
        now = datetime.now()
        with state._lock:
            last_b = state._last_blocked_log[src_ip]
            should_log = (last_b is None or
                          (now - last_b).total_seconds() >= state._blocked_interval)
            if should_log:
                state._last_blocked_log[src_ip] = now
        if should_log:
            log_blocked(src_ip, method, path)
            file_log.warning(f"BLOCKED {src_ip} {method} {path} (banned)")
        else:
            with stats_lock:
                stats["blocked"] += 1  # still count it, just don't print
        return True, 0.0

    features = extract_features(handler, content_length, path)
    score    = float(model.decision_function(features)[0])

    # Use our custom threshold — more negative = more anomalous
    # ANOMALY_SCORE_THRESHOLD = -0.5 means only flag strong anomalies
    is_anomaly = score < ANOMALY_SCORE_THRESHOLD

    if is_anomaly:
        if src_ip in HTTP_PASSTHROUGH or src_ip in WHITELIST:
            log_monitor(src_ip, method, path, score)
            file_log.info(f"MONITOR {src_ip} {method} {path} score={score:+.3f}")
            return False, score
        else:
            state.record_anomaly(src_ip, score, method, path)
            return True, score

    return False, score

# ── Request handler ───────────────────────────────────────────────────────────
class PicoHandler(BaseHTTPRequestHandler):
    timeout = 5  # Drop slow/Slowloris connections after 5s

    def setup(self):
        """Track connection open — called when connection is established."""
        super().setup()
        src_ip = self.client_address[0]
        if src_ip not in WHITELIST and src_ip not in HTTP_PASSTHROUGH:
            count = state.open_connection(src_ip)
            if count > SLOWLORIS_CONN_LIMIT and not state.is_banned(src_ip):
                log_slowloris(src_ip, count)
                file_log.warning(f"SLOWLORIS {src_ip} {count} open connections")
                state.record_anomaly(src_ip, -1.0, 'CONN', '/slowloris')

    def finish(self):
        """Track connection close."""
        src_ip = self.client_address[0]
        state.close_connection(src_ip)
        super().finish()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body           = self.rfile.read(content_length)
        src_ip         = self.client_address[0]
        src_port       = self.client_address[1]

        blocked, score = ids_check(self, content_length, self.path)
        if blocked:
            self._respond(403, 'blocked')
            return

        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._respond(400, 'Bad Request')
            return

        if self.path == '/sensor':
            entry = build_entry(src_ip, src_port, self.server.server_address[1], data, '/sensor')
            log_ok(src_ip, 'POST', '/sensor', score,
                   f"pico={data.get('pico_id','?')}")
            file_log.info(f"OK {src_ip} POST /sensor score={score:+.3f} pico={data.get('pico_id')}")
            save_data(entry)
            self._respond(200, 'ok')

        elif self.path == '/alert':
            entry = build_entry(src_ip, src_port, self.server.server_address[1], data, '/alert')
            log_ok(src_ip, 'POST', '/alert', score,
                   f"pico={data.get('pico_id','?')} status={data.get('status','?')}")
            file_log.warning(f"ALERT {src_ip} POST /alert score={score:+.3f} pico={data.get('pico_id')}")
            save_data(entry)
            self._respond(200, 'ok')

        elif self.path == '/diagnostics':
            entry = build_entry(src_ip, src_port, self.server.server_address[1], data, '/diagnostics')
            log_ok(src_ip, 'POST', '/diagnostics', score,
                   f"mem={data.get('free_mem')} cycle={data.get('cycle')}")
            file_log.info(f"OK {src_ip} POST /diagnostics score={score:+.3f}")
            save_data(entry)
            self._respond(200, 'ok')

        elif self.path == '/join':
            entry = build_entry(src_ip, src_port, self.server.server_address[1], data, '/join')
            log_join(src_ip, data.get('pico_id', '?'))
            file_log.info(f"JOIN {src_ip} pico={data.get('pico_id')} score={score:+.3f}")
            save_data(entry)
            self._respond(200, 'ok')

        elif self.path == '/lora':
            label = int(data.get('label', 0))
            entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'src_ip':    src_ip,
                'ts':        data.get('ts', 0),
                'dt':        data.get('dt', 0),
                'rssi':      data.get('rssi', 0),
                'snr':       data.get('snr', 0),
                'label':     'lora_attack' if label == 1 else 'normal',
            }
            log_lora(src_ip, entry['label'], entry['rssi'], entry['snr'], entry['dt'])
            outfile = 'lora_attack_data.json' if label == 1 else 'lora_normal_data.json'
            with open(outfile, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            if label == 1:
                file_log.warning(f"LORA ATTACK {src_ip} RSSI={entry['rssi']} SNR={entry['snr']}")
            else:
                file_log.info(f"LORA OK {src_ip} RSSI={entry['rssi']} SNR={entry['snr']}")
            self._respond(200, 'ok')

        else:
            self._respond(404, 'Not Found')

    def do_GET(self):
        blocked, score = ids_check(self, 0, self.path)
        if blocked:
            self._respond(403, 'blocked')
            return

        src_ip   = self.client_address[0]
        src_port = self.client_address[1]

        if self.path == '/status':
            log_ok(src_ip, 'GET', '/status', score)
            save_data(build_entry(src_ip, src_port,
                                  self.server.server_address[1], {}, '/status'))
            file_log.info(f"OK {src_ip} GET /status score={score:+.3f}")
            print_separator()
            print_status_bar()
            print_separator()
            self._respond(200, 'IDS Running')

        elif self.path == '/ntp':
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            log_ok(src_ip, 'GET', '/ntp', score)
            file_log.info(f"OK {src_ip} GET /ntp score={score:+.3f}")
            self._respond(200, json.dumps({'status': 'ok', 'time': now}))

        elif self.path == '/':
            log_ok(src_ip, 'GET', '/', score)
            self._respond(200, json.dumps({'status': 'ok', 'service': 'IDS/IPS Pi node'}))

        else:
            self._respond(404, 'Not Found')

    def _respond(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(message.encode())

    def log_message(self, format, *args):
        pass  # suppress default server noise

    def handle_error(self, request, client_address):
        pass  # suppress timeout/connection reset errors


# ── Threaded server for Slowloris protection ──────────────────────────────────
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global model

    parser = argparse.ArgumentParser(description="HTTP IDS/IPS for Raspberry Pi")
    parser.add_argument('--model', default='ids_model.pkl')
    parser.add_argument('--port',  type=int, default=80)
    parser.add_argument('--host',  default='0.0.0.0')
    args = parser.parse_args()

    print_header()

    print(f"  {C.GRAY}Loading model: {args.model} ...{C.RESET}", end=" ", flush=True)
    try:
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['anomaly_detector']
        print(f"{C.GREEN}OK{C.RESET}  {C.GRAY}(trained: {model_data.get('timestamp', 'unknown')}){C.RESET}")
    except FileNotFoundError:
        print(f"{C.RED}ERROR: {args.model} not found.{C.RESET}")
        return
    except KeyError:
        print(f"{C.RED}ERROR: Model file format not recognised.{C.RESET}")
        return

    print(f"\n  {C.GREEN}Listening on {args.host}:{args.port}{C.RESET}")
    print(f"  {C.GRAY}Strike limit : {ANOMALY_THRESHOLD} anomalies before ban{C.RESET}")
    print(f"  {C.GRAY}Ban duration : {BAN_DURATION_SECS}s{C.RESET}")
    print(f"  {C.GRAY}Score thresh : {ANOMALY_SCORE_THRESHOLD} (negative = anomalous){C.RESET}")
    print(f"  {C.GRAY}Whitelist    : {sorted(WHITELIST)}{C.RESET}")
    print(f"  {C.GRAY}Passthrough  : {sorted(HTTP_PASSTHROUGH)}{C.RESET}")
    print()
    print_separator()
    print(
        f"  {C.GRAY}{'Time':>8}   "
        f"{'Status':<10}  "
        f"{'IP':<16}  "
        f"{'Meth':<5}  "
        f"{'Endpoint':<15}  "
        f"{'Score':<12}  "
        f"Details{C.RESET}"
    )
    print_separator()

    httpd = ThreadedHTTPServer((args.host, args.port), PicoHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}IDS stopped.{C.RESET}")
        print_separator()
        print_status_bar()
        print_separator()
        print()


if __name__ == '__main__':
    main()