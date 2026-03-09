#!/usr/bin/env bash
set -euo pipefail

OLD_UID=$(id -u kujira)
OLD_GID=$(id -g kujira)

NEW_UID=${NEW_UID:-1000}
NEW_GID=${NEW_GID:-1000}

if [ -n "$NEW_UID" ] && [ "$NEW_UID" != "$OLD_UID" ]; then
    echo "🔧 Updating 'kujira' user and group IDs to match host system..."
    usermod -u "$NEW_UID" kujira
    find / -user "$OLD_UID" -exec chown -h "$NEW_UID" {} \; 2>/dev/null || true
    echo "✅ 'kujira' user and group IDs updated successfully."
fi

if [ -n "$NEW_GID" ] && [ "$NEW_GID" != "$OLD_GID" ]; then
    echo "🔧 Updating 'kujira' group ID to match host system..."
    groupmod -g "$NEW_GID" kujira
    find / -group "$OLD_GID" -exec chown -h :"$NEW_GID" {} \; 2>/dev/null || true
    echo "✅ 'kujira' group ID updated successfully."
fi

WORKSPACE="${WORKSPACE:-${HOME}}"
if [ -f "${WORKSPACE}/pyproject.toml" ]; then
    gosu kujira bash -c "cd '${WORKSPACE}' && uv sync --frozen"
fi

exec gosu kujira "$@"
