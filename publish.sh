#!/bin/bash
set -o errexit
set -o nounset
cd "$(dirname "$0")"

echo "Deactivating any active virtual environment..."
deactivate 2>/dev/null || true

if [ ! -d dist ]
then
    >&2 echo "No dist/ directory found. Please run build.sh first."
    exit 1
fi

DIST_FILES="$(ls -1 dist/*.tar.gz dist/*.whl)"
if [ -z "$DIST_FILES" ]
then
    >&2 echo "No distribution files found. Please run build.sh first."
    exit 1
fi

# Require that the tests have been run !
if [ ! -e dist/.tested ]
then
    >&2 echo "Please run test-dist.sh before publishing."
    exit 1
else
    # And that no file in src/ is newer than the test marker file
    if [ ! -z "$(find src -type f -newer dist/.tested)" ]
    then
        >&2 echo "Source files have been modified since tests were last run. Please run test-dist.sh again."
        exit 1
    fi
fi

# Get confirmation from user
echo "Found these files to publish:"
echo "$DIST_FILES" | sed 's/^/    /'
read -n1 -r -p "Ready to publish [y/n]? " confirm
echo
if [[ ! "$confirm" =~ ^[Yy]$ ]]
then
    >&2 echo "Aborting publication."
    exit 1
fi

# Create a clean venv for building
if [ -d .venv-publish ]
then
    rm -rf .venv-publish
fi

# Create and activate a publish venv
python3 -m venv .venv-publish
source .venv-publish/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade twine

# And publish.
DIST_FILES_ONE_LINE=$(echo "$DIST_FILES" | tr '\n' ' ')
python3 -m twine upload $DIST_FILES_ONE_LINE

echo
echo "Done. You'll need to reactivate your previous virtual environment if you had one."