#!/bin/bash
set -o errexit
set -o nounset
cd "$(dirname "$0")"

source python.sh

echo "Deactivating any active virtual environment..."
deactivate 2>/dev/null || true

if [ ! -d dist ] || [ -z "$(ls dist/*.whl)" ]
then
    >&2 echo "No dist/*.whl found. Please run build.sh first."
    exit 1
fi

# Remove any previous test marker
rm -f dist/.tested

# Create and activate a clean test venv
rm -rf .venv-test
$PYTHON -m venv .venv-test
source .venv-test/bin/activate
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade dist/*.whl
$PYTHON -m pip install '.[test]'

# And test.
echo -e "\033[1mTesting:\033[0m"
cd tests
pytest -v && touch ../dist/.tested

echo
echo "Done. You'll need to reactivate your previous virtual environment if you had one."
