#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python -m PyInstaller --clean --noconfirm "${PROJECT_ROOT}/packaging/pyinstaller.spec"
