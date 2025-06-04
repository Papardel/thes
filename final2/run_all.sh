#!/usr/bin/env bash
#
# run_all.sh – invoke run.py for each Defects4J bug,
#              skip up to START_FROM/START_BUG, and
#              record & skip any that fail.
#
set -euo pipefail

RUNNER="./pipe.py"
START_FROM=""
START_BUG=""
FAILURES="failures.txt"

usage() {
  echo "Usage: $0 [-s START_PROJECT] [-b START_BUG]"
  exit 1
}

while (( "$#" )); do
  case "$1" in
    -s|--start-from) START_FROM="$2"; shift 2;;
    -b|--start-bug)  START_BUG="$2";  shift 2;;
    -h|--help)       usage;;
    *)               usage;;
  esac
done

touch "$FAILURES"

echo "Fetching all project IDs…"
projects=( $(defects4j pids) )
resume_proj=0

for project in "${projects[@]}"; do
  if [[ -n $START_FROM && $resume_proj -eq 0 ]]; then
    if [[ $project == "$START_FROM" ]]; then
      resume_proj=1
    else
      echo "Skipping project $project"
      continue
    fi
  else
    resume_proj=1
  fi

  echo "=== $project ==="
  skip_bug=0
  [[ -n $START_BUG ]] && skip_bug=1

  for bug in $(defects4j bids -p "$project"); do
    combo="${project}-${bug}"

    # resume logic for START_BUG
    if (( skip_bug )); then
      if [[ $bug == "$START_BUG" ]]; then
        skip_bug=0
      else
        echo "  skipping bug $combo"
        continue
      fi
    fi

    # skip if previously failed
    if grep -Fxq "$combo" "$FAILURES"; then
      echo "  skipping previously FAILED $combo"
      continue
    fi

    echo "---- $combo ----"
    if python3 "$RUNNER" "$project" "$bug" "train full"; then
      echo "  DONE $combo"
    else
      echo "  ** FAILED $combo **"
      # record it
      echo "$combo" >> "$FAILURES"
    fi
  done
done

# ding when done
printf '\a'
