#!/usr/bin/env bash

mkdir -p logs

nohup uv run jupyter lab \
    --no-browser \
    --ip=127.0.0.1 \
    --port=8888 \
    > logs/jupyter.log 2>&1 &

echo "Started Jupyter detached."
echo "See logs using: 'tail -f logs/jupyter.log'"
echo "Remember to select python runtime 'Python (flower-project)' in Jupyter"