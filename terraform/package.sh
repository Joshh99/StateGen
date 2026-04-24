#!/bin/bash
# package.sh — zip the project for upload to Azure Blob.
# Run this once before 'terraform apply'.
#
# Usage:  bash terraform/package.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT="$SCRIPT_DIR/../stategen_code.zip"

echo "Packaging project from: $PROJECT_ROOT"
echo "Output:                  $OUT"

cd "$PROJECT_ROOT"

zip -r "$OUT" . \
  --exclude ".venv/*" \
  --exclude ".git/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude "*.pyo" \
  --exclude ".ipynb_checkpoints/*" \
  --exclude "results/*" \
  --exclude "logs/*" \
  --exclude "memory/*" \
  --exclude "terraform/*" \
  --exclude "stategen_code.zip" \
  --exclude "bigcodebench-new/.git/*" \
  --exclude "bigcodebench/.git/*" \
  --exclude "*.pdf" \
  --exclude ".env" \
  --exclude ".claude/*" \
  --exclude ".idea/*"

SIZE=$(du -sh "$OUT" | cut -f1)
echo "Done. Package size: $SIZE"
echo ""
echo "Next step:"
echo "  cd terraform && terraform apply"
