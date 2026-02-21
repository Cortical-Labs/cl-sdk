#!/bin/bash
set -o errexit
set -o nounset
cd "$(dirname "$0")"

# This file is used to control the Python version used by the build,
# test, and publish scripts. By default it will uses Python 3.12,
# but you can set the PYTHON environment variable to use a different
# version if needed.
#
# Usage:
#   source python.sh

PYTHON=${PYTHON:-python3.12}

# if python is not executable, print an error and exit
if ! command -v $PYTHON &> /dev/null
then
    >&2 echo "Python executable '$PYTHON' not found. Please set the PYTHON environment variable to a suitable Python executable."
    exit 1
fi

$PYTHON --version
