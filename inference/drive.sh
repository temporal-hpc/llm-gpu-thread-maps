#!/bin/bash
# Sends every payload in $PAYLOADS to a local Ollama server and stores the raw
# HTTP reply in $RAW. Pure bash on purpose: the official ollama image ships
# bash and coreutils only, with no curl and no Python, so the request goes
# through /dev/tcp. The replies are decoded later by parse_raw.py.
# Resumable: a payload whose reply already exists is skipped.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAYLOADS="${PAYLOADS:-$ROOT/payloads}"
RAW="${RAW:-$ROOT/raw}"
mkdir -p "$RAW"

post () {  # $1 = json payload, $2 = output file
  local len; len=$(wc -c < "$1")
  exec 3<>/dev/tcp/127.0.0.1/11434 || return 1
  { printf 'POST /api/generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n' "$len"
    cat "$1"; } >&3
  cat <&3 > "$2"
  exec 3<&- 2>/dev/null; exec 3>&- 2>/dev/null
}

# wait for the server
until (exec 3<>/dev/tcp/127.0.0.1/11434) 2>/dev/null; do sleep 3; done
exec 3<&- 2>/dev/null; exec 3>&- 2>/dev/null
echo "server is up"

n=0; t0=$SECONDS
for f in "$PAYLOADS"/*.json; do
  [ -e "$f" ] || { echo "no payloads in $PAYLOADS"; exit 0; }
  b=$(basename "$f" .json)
  [ -s "$RAW/$b.raw" ] && continue
  s=$SECONDS
  post "$f" "$RAW/$b.raw"
  n=$((n+1))
  echo "  $b  $((SECONDS-s))s  (total $n, $(( (SECONDS-t0)/60 )) min)"
done
echo "DRIVE_DONE n=$n"
