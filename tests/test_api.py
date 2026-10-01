import json
import socket
import subprocess
import sys
import time
import urllib.request

BASE_URL = ""

def request(method, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE_URL + path, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, json.loads(response.read())


def main():
    global BASE_URL
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    BASE_URL = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if proc.poll() is not None:
                raise RuntimeError("Test server failed to start")
            try:
                request("GET", "/health")
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError("Test server did not become healthy")
        assert request("GET", "/health")[1]["status"] == "ok"
        assert request("GET", "/ready")[1]["mode"] == "deterministic_demo"
        assert request("GET", "/ready")[1]["model_loaded"] is False
        status, out = request("POST", "/v1/risk-scores", {"records": [{"taxpayer_id": "t-1", "filings_24m": 10, "late_filings_24m": 9, "payment_ratio_12m": .1, "days_since_last_payment": 300}]}, {"Authorization": "Bearer demo-review-token"})
        assert status == 200 and out["results"][0]["risk_band"] == "HIGH"
        assert request("POST", "/v1/assistant", {"question": "What payment channels are supported?"})[1]["grounded"] is True
        assert request("POST", "/v1/complaints", {"text": "My payment is missing"})[1]["category"] == "payment"
        assert request("POST", "/v1/meter-review", {"reading_value": 4, "previous_reading": 9, "ocr_confidence": .9})[1]["requires_review"] is True
        print("API smoke tests passed")
    finally:
        proc.terminate(); proc.wait(timeout=5)


if __name__ == "__main__": main()
