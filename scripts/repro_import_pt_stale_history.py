"""Repro: importing a .pt while stale history exists.

Expected after fix:
- importing .pt deletes training_history_{symbol}.json
- overview current_step matches checkpoint step (not stale history)
"""

from __future__ import annotations

import json
import mimetypes
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from web.training_package import build_training_export_zip

API = "http://127.0.0.1:8765"
SYMBOL = "ADAUSD"


def main() -> None:
    # Ensure stale history exists (do not modify its content here)
    hist = Path("training_history_ADAUSD.json")
    print("history exists:", hist.exists())

    body, name = build_training_export_zip(SYMBOL)
    zip_path = Path(tempfile.gettempdir()) / name
    zip_path.write_bytes(body)
    print("zip:", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        pt_member = [n for n in zf.namelist() if n.endswith(".pt")][0]
        pt_bytes = zf.read(pt_member)

    pt_path = Path(tempfile.gettempdir()) / Path(pt_member).name
    pt_path.write_bytes(pt_bytes)
    print("pt:", pt_path)

    boundary = "----t"
    ctype = mimetypes.guess_type(pt_path.name)[0] or "application/octet-stream"
    payload = (
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{pt_path.name}\"\r\n"
         f"Content-Type: {ctype}\r\n\r\n").encode()
        + pt_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        API + f"/api/training/import?symbol={SYMBOL}",
        data=payload,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    print("import resp:", resp)

    ov = json.loads(urllib.request.urlopen(API + "/api/overview").read())
    print("overview current_step:", (ov.get("progress") or {}).get("current_step"))
    print("history exists after import:", hist.exists())


if __name__ == "__main__":
    main()

