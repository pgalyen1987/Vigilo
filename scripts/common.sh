#!/usr/bin/env bash
# Shared helpers for Vigilo capture scripts. Source from scripts/*.sh:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Resolves the repo root from the script location (portable — no hardcoded paths).

_vigilo_script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
VIGILO="$(cd "$_vigilo_script_dir/.." && pwd)"
VIGILO_IMAGE="${VIGILO_IMAGE:-vigilo:latest}"

if [ -f "$VIGILO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$VIGILO/.env"
    set +a
fi
