#!/bin/bash
set -o errexit
set -o nounset
cd "$(dirname "$0")"

echo "Deactivating any active virtual environment..."
deactivate 2>/dev/null || true

# Create and activate a clean build venv
rm -rf .venv-build dist
python3 -m venv .venv-build
source .venv-build/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build

# And build. This recreates dist/*
rm -rf dist
python3 -m build
deactivate

# Done. List the files in the tarball.
echo -e "\033[1mCreated:\033[0m"
find dist -mindepth 1 | sed 's/^/    /'

echo -e "\033[1mTarball contents:\033[0m"
tar tzf dist/*.tar.gz | sed 's/^/    /'

echo -e "\033[1mWheel contents:\033[0m"
unzip -Z1 dist/*.whl | sed 's/^/    /'

echo
echo "Done. You'll need to reactivate your previous virtual environment if you had one."