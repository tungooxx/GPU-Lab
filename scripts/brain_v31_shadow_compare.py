"""Read-only comparison of frozen BrainBench policies, including v3.1.

This intentionally runs no MCP action and creates no ResearchDecision.  It is
an inspectable policy diff, not evidence that a policy is scientifically true.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gpu_lab.brain_bench import BenchmarkPolicy, ResearchBrainBench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("research_bench"))
    arguments = parser.parse_args()
    bench = ResearchBrainBench(arguments.root)
    rows = []
    for episode in bench.load_all():
        old = bench.baseline_decision(episode, BenchmarkPolicy.BRAIN_V2_STRATEGY_AUGMENTED)
        new = bench.baseline_decision(episode, BenchmarkPolicy.BRAIN_V3_1_DISCOVERY_SEARCH)
        rows.append(
            {
                "episode": episode.episode_id,
                "v2_selected": old.selected_action_id,
                "v31_selected": new.selected_action_id,
                "v31_context": episode.v31_context,
                "changed": old.selected_action_id != new.selected_action_id,
            }
        )
    for row in rows:
        print(
            f"{row['episode']}: v2={row['v2_selected']} -> v3.1={row['v31_selected']} "
            f"changed={row['changed']} context={row['v31_context']}"
        )
    print(f"READ_ONLY_SHADOW episodes={len(rows)} changed={sum(row['changed'] for row in rows)}")


if __name__ == "__main__":
    main()
