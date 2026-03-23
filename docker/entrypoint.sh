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
    EXISTING_GROUP_NAME="$(getent group "$NEW_GID" | cut -d: -f1 || true)"

    # On macOS, GID 20 (the staff group) is commonly used and may already exist in Ubuntu environments.
    # This fallback is prepared to avoid errors when running docker compose.
    if [ -n "$EXISTING_GROUP_NAME" ] && [ "$EXISTING_GROUP_NAME" != "kujira" ]; then
        echo "ℹ️ GID '$NEW_GID' already belongs to '$EXISTING_GROUP_NAME'. Reusing this group for 'kujira'."
        usermod -g "$NEW_GID" kujira
    else
        groupmod -g "$NEW_GID" kujira
    fi

    find / -group "$OLD_GID" -exec chown -h :"$NEW_GID" {} \; 2>/dev/null || true
    echo "✅ 'kujira' group ID updated successfully."
fi

WORKSPACE="${WORK_DIR:-${HOME}}"
if [ -f "${WORKSPACE}/pyproject.toml" ]; then
    gosu kujira bash -c "cd '${WORKSPACE}' && uv sync --frozen"
fi

exec gosu kujira "$@"
