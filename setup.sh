#!/usr/bin/env bash
set -e

echo "Syncing dependencies..."
uv sync
echo "Dependencies synced!"

echo "Registering Jupyter kernel..."
uv run python -m ipykernel install --user \
  --name flower-project \
  --display-name "Python (flower-project)"

echo "Setup complete!"