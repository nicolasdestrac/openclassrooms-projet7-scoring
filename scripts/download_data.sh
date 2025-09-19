#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip -o "data/raw/home-credit-default-risk.zip" -d data/raw
