import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8766"
src = None
for base in Path("D:/").iterdir():
    for p in base.glob("SATSUSDT_M5.parquet"):
        src = str(p)

body = json.dumps({"data_file": src}).encode()
req = urllib.request.Request(
    API + "/api/training/start",
    data=body,
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    print("start", r.read().decode()[:200])

time.sleep(2)
st = json.loads(urllib.request.urlopen(API + "/api/training/status").read())
print("active", st.get("active"), "state", st.get("job", {}).get("state"))

from web.training_package import build_training_export_zip

data, name = build_training_export_zip("SATSUSDT")
boundary = "----test"
payload = (
    (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n\r\n").encode()
    + data
    + f"\r\n--{boundary}--\r\n".encode()
)
req = urllib.request.Request(
    API + "/api/training/import?symbol=SATSUSDT",
    data=payload,
    method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
try:
    urllib.request.urlopen(req, timeout=15)
    print("FAIL: import should be blocked")
except urllib.error.HTTPError as e:
    print("import blocked", e.code, e.read().decode()[:160])

urllib.request.urlopen(urllib.request.Request(API + "/api/training/stop", method="POST"))
print("stopped")
