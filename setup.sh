#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
uv venv

echo "Activating environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing project..."
uv pip install -e .

echo "Registering Jupyter kernel..."
python -m ipykernel install --user \
  --name flower-project \
  --display-name "Python (flower-project)"

echo "Setup complete!"
echo "Now select 'Python (flower-project)' in Jupyter"
