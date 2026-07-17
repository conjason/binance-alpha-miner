"""Import a given training zip via API and print resulting overview step."""

from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8765"
ZIP_PATH = Path(r"C:\Users\Administrator\Downloads\training_ADAUSD_step0060.zip")


def main() -> None:
    data = ZIP_PATH.read_bytes()
    boundary = "----t"
    ctype = mimetypes.guess_type(ZIP_PATH.name)[0] or "application/zip"
    payload = (
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{ZIP_PATH.name}\"\r\n"
         f"Content-Type: {ctype}\r\n\r\n").encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        API + "/api/training/import?symbol=ADAUSD",
        data=payload,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
        print("import resp:", resp)
    except urllib.error.HTTPError as e:
        print("import http", e.code, e.read().decode(errors="replace"))
        return

    ov = json.loads(urllib.request.urlopen(API + "/api/overview").read())
    print("overview step:", (ov.get("progress") or {}).get("current_step"))


if __name__ == "__main__":
    main()

