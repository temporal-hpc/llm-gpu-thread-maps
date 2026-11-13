#!/usr/bin/env python3
"""Turn the raw HTTP replies saved by drive.sh into response files.

drive.sh talks to the Ollama server through /dev/tcp and stores each reply
verbatim (headers + chunked body), because the Ollama container image has
neither curl nor Python. This script, run afterwards on any machine with
Python, decodes the body and writes the "response" field of the JSON to
responses/<cell>.txt, without touching its content.

Usage: python3 inference/parse_raw.py [--raw raw] [--out responses]
"""
import argparse, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join(ROOT, "raw"))
ap.add_argument("--out", default=os.path.join(ROOT, "responses"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def body_of(blob):
    head, _, rest = blob.partition(b"\r\n\r\n")
    if b"chunked" not in head.lower():
        return rest
    out = b""
    while rest:
        size, _, rest = rest.partition(b"\r\n")
        n = int(size.split(b";")[0].strip() or b"0", 16)
        if n == 0:
            break
        out += rest[:n]
        rest = rest[n + 2:]
    return out


ok = bad = 0
for path in sorted(glob.glob(os.path.join(a.raw, "*.raw"))):
    base = os.path.basename(path)[:-4]
    try:
        reply = json.loads(body_of(open(path, "rb").read()))
        if "error" in reply:
            raise ValueError(reply["error"])
        text = reply.get("response", "")
    except Exception as e:
        bad += 1
        print("FAILED %s: %r" % (base, e))
        continue
    with open(os.path.join(a.out, base + ".txt"), "w") as f:
        f.write(text)
    ok += 1
print("%d responses written, %d failed" % (ok, bad))
