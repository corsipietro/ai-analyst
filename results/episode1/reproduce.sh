#!/bin/bash
# Rerun the 6 setups of episode 1 (2 models x 3 strategies) on the 5 domains, 2 seeds per question.
# The exact settings of each run are also recorded in results/episode1/runs/*/runs.jsonl ("overrides").
# Requires: ollama pull gemma4:12b gemma4:26b. Several hours on a laptop. New runs go to bench/runs/.
# Run from the repository root:  bash results/episode1/reproduce.sh
# (Plain bash 3.2 syntax, so it also runs with the bash that ships with macOS.)
set -u
DOMAINS="toy_shell gen_rentals gen_app gen_energy gen_weather"
EXPLORER_THINK="--set roles.explorer.reasoning=true --set roles.explorer.temperature=1.0 --set roles.explorer.top_k=64 --set roles.explorer.top_p=0.95 --set roles.explorer.max_tokens=8192"

setup_args() {
  case "$1" in
    g5_react)      echo "--set graph=v2 --set prompt=v3 $EXPLORER_THINK" ;;
    g5_rewoo)      echo "--set graph=rewoo --set prompt=v4" ;;
    g5_repair)     echo "--set graph=rewoo_repair --set prompt=v4" ;;
    g5_react_26b)  echo "--set graph=v2 --set prompt=v3 --set roles.explorer.model=gemma4:26b --set roles.interpreter.model=gemma4:26b $EXPLORER_THINK" ;;
    g5_rewoo_26b)  echo "--set graph=rewoo --set prompt=v4 --set roles.planner.model=gemma4:26b --set roles.analyst.model=gemma4:26b" ;;
    g5_repair_26b) echo "--set graph=rewoo_repair --set prompt=v4 --set roles.planner.model=gemma4:26b --set roles.analyst.model=gemma4:26b" ;;
  esac
}

for tag in g5_react g5_rewoo g5_repair g5_react_26b g5_rewoo_26b g5_repair_26b; do
  for d in $DOMAINS; do
    echo "=== $(date '+%H:%M') $tag $d"
    python -m bench.run_bench --domain "$d" --reps 2 $(setup_args "$tag") --tag "$tag"
  done
done
echo "=== $(date '+%H:%M') done"
