from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from datetime import datetime

LOG_FILE = "sensor_data.json"

logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def save_data(data):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

def build_entry(handler, data, endpoint):
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['src_ip']    = handler.client_address[0]
    data['src_port']  = handler.client_address[1]
    data['dst_port']  = handler.server.server_address[1]
    data['endpoint']  = endpoint
    data['label']     = "normal"
    return data

class PicoHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except:
            self._respond(400, "Bad Request")
            return

        if self.path == '/sensor':
            entry = build_entry(self, data, "/sensor")
            print(f"[{entry['timestamp']}] {entry['src_ip']}:{entry['src_port']} -> Pi:{entry['dst_port']} | {data.get('pico_id')} | /sensor | Temp: {data.get('temperature')} Humidity: {data.get('humidity')}")
            save_data(entry)
            self._respond(200, "OK")

        elif self.path == '/alert':
            entry = build_entry(self, data, "/alert")
            print(f"[{entry['timestamp']}] ⚠ ALERT from {data.get('pico_id')} | Temp: {data.get('temperature')} Status: {data.get('status')}")
            save_data(entry)
            self._respond(200, "OK")

        elif self.path == '/diagnostics':
            entry = build_entry(self, data, "/diagnostics")
            print(f"[{entry['timestamp']}] {data.get('pico_id')} | /diagnostics | Free mem: {data.get('free_mem')} Cycle: {data.get('cycle')}")
            save_data(entry)
            self._respond(200, "OK")

        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unknown POST to {self.path} from {self.client_address[0]}")
            self._respond(404, "Not Found")

    def do_GET(self):
        if self.path == '/status':
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip":    self.client_address[0],
                "src_port":  self.client_address[1],
                "dst_port":  self.server.server_address[1],
                "endpoint":  "/status",
                "label":     "normal"
            }
            print(f"[{entry['timestamp']}] {entry['src_ip']}:{entry['src_port']} -> Pi:{entry['dst_port']} | GET /status")
            save_data(entry)
            self._respond(200, "Pi Server Running")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unknown GET to {self.path} from {self.client_address[0]}")
            self._respond(404, "Not Found")

    def _respond(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(message.encode())

    def log_message(self, format, *args):
        logging.info(f"{self.client_address[0]} - {format % args}")

print("Pi server starting on port 80...")
print("-" * 55)
httpd = httpd = HTTPServer(('10.42.0.1', 80), PicoHandler)
httpd.serve_forever()