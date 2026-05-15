"""
r_pico_lora.py — LoRa Receiver Pico
Polls for packets and forwards valid ones to Pi lora_server.py via HTTP POST.
"""
import time
import json
import urequests
from sx1262 import SX1262

PI_URL        = "http://10.42.0.1/lora"
WIFI_SSID     = "Wifi-SSID"      # Pi hotspot SSID
WIFI_PASSWORD = "Wifi-Password"                   # Pi hotspot password (blank if none)

# ── Connect to Pi Wi-Fi ───────────────────────────────────────────────────────
import network
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
if not wlan.isconnected():
    print("Connecting to Wi-Fi...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    timeout = 15
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1
    if wlan.isconnected():
        print("Wi-Fi connected! IP:", wlan.ifconfig()[0])
    else:
        print("Wi-Fi failed — check SSID/password")
        raise SystemExit

sx = SX1262(spi_bus=1, clk=10, mosi=11, miso=12, cs=3, irq=20, rst=15, gpio=2)

sx.begin(freq=923, bw=125.0, sf=9, cr=8, syncWord=0x12,
         power=-5, currentLimit=60.0, preambleLength=8,
         implicit=False, implicitLen=0xFF,
         crcOn=True, txIq=False, rxIq=False,
         tcxoVoltage=1.7, useRegulatorLDO=False, blocking=True)

print("Receiver started. Listening for packets...")

last_arrival = [None]

while True:
    # blocking=True means recv() waits until a packet arrives
    msg, err = sx.recv()

    if err != 0 or not msg or len(msg) < 5:
        continue

    arrival_local = time.time()

    # ── Extract RSSI/SNR ──────────────────────────────────────────────────────
    # getPacketStatus() returns a packed 32-bit int on this library version
    # SX1262 datasheet: rssi = -byte0/2, snr = byte1/4 (signed)
    status = sx.getPacketStatus()
    if isinstance(status, int):
        b       = status.to_bytes(4, 'little')
        hw_rssi = int(-b[0] / 2)
        snr_raw = b[1]
        hw_snr  = (snr_raw - 256 if snr_raw > 127 else snr_raw) / 4
    elif isinstance(status, (list, tuple)):
        hw_rssi, hw_snr = status[0], status[1]
    elif isinstance(status, dict):
        hw_rssi = status.get('rssi', -100)
        hw_snr  = status.get('snr',  0)
    else:
        hw_rssi, hw_snr = -100, 0

    # ── Decode and validate ───────────────────────────────────────────────────
    try:
        raw   = msg.decode('utf-8').strip()
        parts = raw.split(',')
    except Exception:
        continue

    if len(parts) < 7 or parts[0] != "START":
        print("Invalid: {}".format(raw[:40]))
        continue

    try:
        seq              = int(parts[1])
        transmitted_real = float(parts[2])
        transmitted_sent = float(parts[3])
        sim_rssi         = int(parts[4])
        sim_snr          = float(parts[5])
        label            = int(parts[6])
    except Exception as e:
        print("Parse error: {}".format(e))
        continue

    # ── Inter-packet delay ────────────────────────────────────────────────────
    ipd = 0.0 if last_arrival[0] is None else arrival_local - last_arrival[0]
    last_arrival[0] = arrival_local

    print("RX SEQ:{} RSSI:{} SNR:{} IPD:{:.2f} label:{}".format(
        seq, hw_rssi, hw_snr, ipd, label))

    # ── Build and POST entry ──────────────────────────────────────────────────
    entry = {
        "metadata": {"seq": seq, "node_id": "Pico_Node_01"},
        "timestamps": {
            "transmitted_real": transmitted_real,
            "transmitted_sent": transmitted_sent,
            "arrival_local":    arrival_local,
            "delta_t":          transmitted_sent - arrival_local
        },
        "physical_layer": {
            "hw_rssi": hw_rssi, "hw_snr": hw_snr,
            "sim_rssi": sim_rssi, "sim_snr": sim_snr
        },
        "ipd":   ipd,
        "label": label
    }

    try:
        r = urequests.post(
            PI_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(entry)
        )
        r.close()
    except Exception as e:
        print("POST failed: {}".format(e))
