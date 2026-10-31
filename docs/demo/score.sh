#!/bin/sh
# Score one email file with the mailguard API and print the verdict with its strongest reasons.
#
#   ./score.sh phishing.eml        # needs curl and jq, and the API (make demo) on port 8000
set -eu

jq -Rs '{raw: .}' "$1" \
  | curl -sf "${MAILGUARD_URL:-http://127.0.0.1:8000}/v1/predict" \
      -H 'content-type: application/json' -d @- \
  | jq -r '
      def r2: . * 100 | round / 100;
      def colour: (if . == "phishing" then "\u001b[1;31m" else "\u001b[1;32m" end) + . + "\u001b[0m";
      "label        \(.label | colour)",
      "probability  \(.probability * 1000 | round / 10)% in a mailbox with \(.base_rate * 100)% phishing",
      "model        \(.model.name) v\(.model.version), threshold \(.model.threshold | r2)",
      "reasons",
      (.reasons[] | "  \(if .contribution > 0 then "+" else "" end)\(.contribution | r2)\t\(.group)/\(.feature | sub("^[a-z]+__"; ""))")
    '
