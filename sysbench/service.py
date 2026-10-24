"""Fixed-work HTTP target; calibration and test use the same CPU work."""
import hashlib
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

POD_NAME = os.environ.get("HOSTNAME", "unknown")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            payload = b"ready\n"
        elif self.path == "/work":
            payload = hashlib.pbkdf2_hmac("sha256", b"paper4-fixed-request", b"fixed-salt", 6000).hex().encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Type", "text/plain")
        self.send_header("X-Paper4-Pod", POD_NAME)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    delay = float(os.environ.get("STARTUP_SLEEP", "5"))
    if not 0 <= delay <= 120:
        raise ValueError("Invalid startup delay")
    time.sleep(delay)
    print("Fixed CPU work: pbkdf2_hmac sha256 iterations=6000; startup sleep=", delay, flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
