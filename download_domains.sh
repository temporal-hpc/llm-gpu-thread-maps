#!/bin/bash
# Downloads the six ground-truth domain files (10^6 points each, 250 MB in
# total once decompressed) from the release of this repository into domains/.
set -eu
REPO="temporal-hpc/llm-gpu-thread-maps"
TAG="v1.0"
BASE="https://github.com/$REPO/releases/download/$TAG"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/domains"
mkdir -p "$DIR"

while read -r sum name; do
  if [ ! -s "$DIR/$name" ]; then
    echo "downloading $name"
    curl -fL --retry 3 -o "$DIR/$name.gz" "$BASE/$name.gz"
    gunzip -f "$DIR/$name.gz"
  fi
  echo "$sum  $DIR/$name" | sha256sum -c -
done <<'SUMS'
94c51a50bada407b172fa878be9871672e3781662b21e7cb3ea2697911898e0c 2D_Alfombra_Sierpinski.txt
3de513b27b3a40b5e961d083f8391938c27372e9b892b1c45df05604d987a053 2D_Triangular.txt
26eb49547b0841e24a6e856899266057ab1ef5fc9657c1113cb176a36a93d236 2D_Triangulo_Sierpinski.txt
1f3f136841b538e29d4f945450a58603b96b91aef75084acd33156ada73d7d93 3D_Menger_Sponge.txt
5844cbe00c0d761c9127fd99602e661f4da6095cfe3a7f1cef9ec114d84c4074 3D_Piramide_Sierpinski.txt
2a5c9f73b2e07a9f37a747dfc177b63cd67a5f120724d2cf93cbbf10a053bbc6 3D_Triangular.txt
SUMS
echo "domains ready in $DIR"
