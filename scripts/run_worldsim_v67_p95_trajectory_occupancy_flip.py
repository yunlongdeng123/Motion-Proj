"""Run the development trajectory-conditioned occupancy-flip experiment."""

import json
import sys
from pathlib import Path

from scripts.run_worldsim_v67_p90_plain_trajectory_max_error import main


if __name__ == "__main__":
    main()
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    runs_root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    summary_path = runs_root / "worldsim_v67" / "WS-V67-P95-TRAJECTORY-OCCUPANCY-FLIP-01" / run_id / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["fresh_test_evaluation"]
    metrics["all_trajectory_occupancy_flip_prevalence"] = metrics["all_unreliable_prevalence"]
    metrics["query_selected_occupancy_flip_prevalence"] = metrics["query_selected_unreliable_prevalence"]
    metrics["actor_selected_occupancy_flip_prevalence"] = metrics["actor_selected_unreliable_prevalence"]
    metrics["frozen_p75_selected_occupancy_flip_prevalence"] = metrics["frozen_p75_selected_unreliable_prevalence"]
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
