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
    # the table view truncates columns ("ephemera"), so read the JSON: an app is finished once its row says stopped
    alive=$($MODAL app list --json 2>/dev/null | python3 -c '
import json, sys
rows = json.load(sys.stdin); app = sys.argv[1]
for r in rows:
    if app in json.dumps(r):
        print(0 if any(str(v).lower() == "stopped" for v in r.values()) else 1); break
else:
    print(0)' "$id" 2>/dev/null || echo 1)
    [ "$alive" = "1" ] && running=$((running + 1))
  done
  if [ "$running" -eq 0 ]; then echo "$(date '+%F %T') all apps finished; guard exits" >> "$LOG"; exit 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    for id in "$@"; do $MODAL app stop -y "$id" >> "$LOG" 2>&1; done
    echo "$(date '+%F %T') DEADLINE: stopped $*" >> "$LOG"; exit 0
  fi
  sleep 300
done
