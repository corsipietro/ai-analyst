#!/bin/bash
# Rerun the 6 setups of episode 2 (2 models x 3 strategies, g5.1 tools) on the 5 domains, 2 seeds per question.
# Same settings as episode 1; the exact settings of each run are also recorded in results/episode2/runs/*/runs.jsonl.
# Requires: ollama pull gemma4:12b gemma4:26b. About 20 hours on a laptop. New runs go to bench/runs/.
# Run from the repository root:  bash results/episode2/reproduce.sh   (or one setup: ... reproduce.sh g51_rewoo_12b)
# (Plain bash 3.2 syntax, so it also runs with the bash that ships with macOS.)
set -u
DOMAINS="toy_shell gen_rentals gen_app gen_energy gen_weather"
EXPLORER_THINK="--set roles.explorer.reasoning=true --set roles.explorer.temperature=1.0 --set roles.explorer.top_k=64 --set roles.explorer.top_p=0.95 --set roles.explorer.max_tokens=8192"

setup_args() {
  case "$1" in
    g51_react_12b)  echo "--set graph=v2 --set prompt=v3 $EXPLORER_THINK" ;;
    g51_rewoo_12b)  echo "--set graph=rewoo --set prompt=v4" ;;
    g51_repair_12b) echo "--set graph=rewoo_repair --set prompt=v4" ;;
    g51_react_26b)  echo "--set graph=v2 --set prompt=v3 $EXPLORER_THINK --set roles.explorer.model=gemma4:26b --set roles.interpreter.model=gemma4:26b" ;;
    g51_rewoo_26b)  echo "--set graph=rewoo --set prompt=v4 --set roles.planner.model=gemma4:26b --set roles.analyst.model=gemma4:26b" ;;
    g51_repair_26b) echo "--set graph=rewoo_repair --set prompt=v4 --set roles.planner.model=gemma4:26b --set roles.analyst.model=gemma4:26b" ;;
    *) echo "unknown setup: $1" >&2; exit 1 ;;
  esac
}

TAGS="${*:-g51_react_12b g51_rewoo_12b g51_repair_12b g51_react_26b g51_rewoo_26b g51_repair_26b}"
for tag in $TAGS; do
  args=$(setup_args "$tag") || exit 1
  for d in $DOMAINS; do
    echo "=== $(date '+%H:%M') $tag $d"
    python -m bench.run_bench --domain "$d" --reps 2 $args --tag "$tag"
  done
done
echo "=== $(date '+%H:%M') done"
