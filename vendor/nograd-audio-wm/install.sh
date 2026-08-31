#!/usr/bin/env bash
# Install script for the wmar_audio artifact.
#
# Usage:
#   bash install.sh
# or with an explicit python:
#   PYTHON=/path/to/python bash install.sh
#
# Two-step install is required because pyworld==0.3.4 must be built against the
# already-installed numpy 1.x (its source distribution does not cap its build
# dependency on numpy, so pip's default build isolation would pull in numpy 2.x,
# whose C headers are incompatible with pyworld 0.3.4).

set -e

PYTHON="${PYTHON:-python}"
PIP="$PYTHON -m pip"

echo "=== Step 1: install numpy and setuptools (needed before pyworld build) ==="
$PIP install "setuptools>=68,<81" "numpy>=1.26,<2.0"

echo "=== Step 2: build pyworld against the installed numpy (no build isolation) ==="
$PIP install --no-build-isolation "pyworld==0.3.4"

echo "=== Step 3: install openai-whisper without build isolation ==="
$PIP install --no-build-isolation "openai-whisper==20250625"

echo "=== Step 4: install remaining requirements ==="
$PIP install -r requirements.txt

echo "=== Done ==="
