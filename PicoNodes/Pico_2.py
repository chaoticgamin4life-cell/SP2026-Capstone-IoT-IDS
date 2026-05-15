import network
import socket
import time
import json
import random

PI_IP = "10.42.0.1"
PI_PORT = 80
PICO_ID = "Pico-2"        # Change for each Pico
PICO_TYPE = "motion"  # Change per Pico: "temperature", "humidity", "motion"

def connect(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    print('Connecting to ' + ssid)
    while not wlan.isconnected():
        print('Waiting...')
        time.sleep(1)
    ip = wlan.ifconfig()[0]
    print('Connected! IP: ' + ip)
    return ip

def read_sensors():
    base = {
        "pico_id": PICO_ID,
        "uptime": time.ticks_ms() // 1000,
        "status": "OK"
    }
    if PICO_TYPE == "temperature":
        temp = round(random.uniform(20.0, 30.0), 2)
        base["temperature"] = temp
        base["humidity"]    = round(random.uniform(40.0, 70.0), 2)
        if temp > 28.0:
            base["status"] = "WARN"
    elif PICO_TYPE == "humidity":
        base["humidity"]  = round(random.uniform(30.0, 90.0), 2)
        base["soil_moisture"] = round(random.uniform(10.0, 80.0), 2)
    elif PICO_TYPE == "motion":
        base["motion_detected"] = random.choice([True, False])
        base["light_level"]     = round(random.uniform(0.0, 100.0), 2)
    return base

def send_request(method, path, body=None):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((PI_IP, PI_PORT))
        if method == "POST" and body:
            body_str = json.dumps(body)
            request = (
                "POST " + path + " HTTP/1.1\r\n"
                "Host: " + PI_IP + "\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + str(len(body_str)) + "\r\n"
                "Connection: close\r\n\r\n" +
                body_str
            )
        else:
            request = (
                "GET " + path + " HTTP/1.1\r\n"
                "Host: " + PI_IP + "\r\n"
                "Connection: close\r\n\r\n"
            )
        s.send(request.encode())
        response = s.recv(1024)
        s.close()
        print(method + " " + path + " -> " + response.decode().split('\r\n')[0])
    except Exception as e:
        print("Send error: " + str(e))

def run():
    cycle = 0
    while True:
        # Always send sensor data
        data = read_sensors()

        # Send alert if status is WARN
        if data.get("status") == "WARN":
            send_request("POST", "/alert", data)
        else:
            send_request("POST", "/sensor", data)

        # Every 5 cycles send a health check GET
        if cycle % 5 == 0:
            send_request("GET", "/status")

        # Every 10 cycles send a full diagnostics POST
        if cycle % 10 == 0:
            diag = {
                "pico_id":    PICO_ID,
                "free_mem":   random.randint(100000, 200000),
                "cpu_freq":   125,
                "ip":         PI_IP,
                "cycle":      cycle
            }
            send_request("POST", "/diagnostics", diag)

        cycle += 1
        # Random interval between 5 and 20 seconds
        time.sleep(random.uniform(5, 20))

while True:
    try:
        connect('Wifi-SSID', 'Wifi-Password')
        run()
    except Exception as e:
        print('Critical error, restarting in 5s: ' + str(e))
        time.sleep(5)

