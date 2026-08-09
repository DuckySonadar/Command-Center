#!/bin/bash
# Give this session the repository owner's git identity.
#
# A Claude Code session in the cloud starts with the tool's own default
# identity (Claude <noreply@anthropic.com>), so every commit it makes is
# authored by a bot. The work is directed by the owner and the copyright is
# the owner's, so the author line should say so. Claude stays on the commit
# as a Co-Authored-By trailer, which is the accurate record of who helped.
#
# Scope, deliberately:
#   - remote sessions only. A local session uses whatever git identity the
#     machine already has, and silently rewriting that would be rude.
#   - `git config --local`, never --global. It touches this clone and
#     nothing else, and `git config --local --unset user.name` undoes it.
#
# To hand the repo to someone else, change these two lines or delete the
# hook; nothing else depends on it.
set -euo pipefail

NAME="DuckySonadar"
EMAIL="77309815+DuckySonadar@users.noreply.github.com"

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "local session -- leaving the git identity alone"
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
git config --local user.name  "$NAME"
git config --local user.email "$EMAIL"
echo "git author set to $NAME <$EMAIL>"
