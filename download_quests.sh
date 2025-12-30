#!/bin/bash
# Download Space Rangers 2 quest files from the online player repository
# Source: https://gitlab.com/vasiliy0/vasiliy0.gitlab.io
#
# Usage:
#   ./download_quests.sh          # Downloads to assets/
#   ./download_quests.sh custom/   # Downloads to custom/

set -e

# Target directory (default: assets/)
TARGET_DIR="${1:-assets}"

echo "=== Space Rangers Quest Downloader ==="
echo "Target directory: $TARGET_DIR"
echo ""

# Create directories
mkdir -p "$TARGET_DIR/qm"
mkdir -p "$TARGET_DIR/json"

# Temporary directory for download
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "Downloading quest files from GitLab..."
curl -sL "https://gitlab.com/vasiliy0/vasiliy0.gitlab.io/-/archive/master/vasiliy0.gitlab.io-master.tar.gz?path=borrowed/qm/SR%202.1.2121%20eng" \
    -o "$TMP_DIR/quests.tar.gz"

echo "Extracting..."
tar -xzf "$TMP_DIR/quests.tar.gz" -C "$TMP_DIR"

# Find and copy .qm files
QM_SOURCE="$TMP_DIR/vasiliy0.gitlab.io-master-borrowed-qm-SR 2.1.2121 eng/borrowed/qm/SR 2.1.2121 eng"
if [ -d "$QM_SOURCE" ]; then
    cp "$QM_SOURCE"/*.qm "$TARGET_DIR/qm/"
    QM_COUNT=$(ls -1 "$TARGET_DIR/qm"/*.qm 2>/dev/null | wc -l | tr -d ' ')
    echo "Copied $QM_COUNT .qm files to $TARGET_DIR/qm/"
else
    echo "Error: Could not find quest files in archive"
    exit 1
fi

echo ""
echo "Converting to JSON format..."
if python3 -c "import quest_evals" 2>/dev/null; then
    python3 -m quest_evals.convert "$TARGET_DIR/qm/" -o "$TARGET_DIR/json/"
    JSON_COUNT=$(ls -1 "$TARGET_DIR/json"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "Converted $JSON_COUNT quests to JSON"
else
    echo "Warning: quest_evals not installed, skipping JSON conversion"
    echo "Install with: pip install -e ."
    echo "Then convert: python -m quest_evals.convert $TARGET_DIR/qm/ -o $TARGET_DIR/json/"
fi

echo ""
echo "=== Done ==="
echo "Quest files are ready in $TARGET_DIR/"
echo ""
echo "To run an evaluation:"
echo "  python -m quest_evals.cli $TARGET_DIR/json/Muzon_eng.json --model claude-sonnet-4.5"
