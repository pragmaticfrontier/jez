#!/usr/bin/env bash
# Stop Modal apps that are still running at a deadline, so a hung job cannot bill overnight. Exits by itself once every
# listed app has stopped. Run detached:
#   setsid nohup tools/modal_deadline_guard.sh "<HH:MM or date>" runs/x.log ap-... ap-... > /dev/null 2>&1 & disown
set -u
cd "$(dirname "$0")/.."
DEADLINE=$(date -d "$1" +%s) || exit 2
LOG=$2; shift 2
MODAL=${MODAL:-.venv/bin/modal}
echo "$(date '+%F %T') guard started; deadline $(date -d @"$DEADLINE" '+%F %T'); apps: $*" >> "$LOG"
while true; do
  running=0
  for id in "$@"; do
    state=$($MODAL app list 2>/dev/null | awk -v id="$id" '$0 ~ id {print $0}' | grep -c -E "running|ephemeral|deployed")
    [ "$state" -gt 0 ] && running=$((running + 1))
  done
  if [ "$running" -eq 0 ]; then echo "$(date '+%F %T') all apps finished; guard exits" >> "$LOG"; exit 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    for id in "$@"; do $MODAL app stop -y "$id" >> "$LOG" 2>&1; done
    echo "$(date '+%F %T') DEADLINE: stopped $*" >> "$LOG"; exit 0
  fi
  sleep 300
done
