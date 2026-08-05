#!/usr/bin/env bash
set -euo pipefail

TH105_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TH105_BUILD_DIR="$TH105_REPO_DIR/build"
mkdir -p "$TH105_BUILD_DIR"

x86_64-w64-mingw32-g++ \
  -std=c++17 -O3 -Wall -Wextra -Werror \
  -shared -static -static-libgcc -static-libstdc++ \
  "$TH105_REPO_DIR/scripts/th105/kernels/hazard.cpp" \
  -o "$TH105_BUILD_DIR/th105_hazard.dll"

sha256sum "$TH105_BUILD_DIR/th105_hazard.dll"
