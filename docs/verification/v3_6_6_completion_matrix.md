# v3.6.6 completion matrix

This matrix is an implementation audit, not a claim that every requested
product behavior is complete.

| Requirement | Current evidence | State |
| --- | --- | --- |
| Separate design, implementation-fidelity, and interpretation records | `correction_v366.py`; PostgreSQL correction suite | Implemented |
| Independent, peer-isolated design attacks and no-majority synthesis | attack membership tables and tests | Implemented |
| Material challenge blocks design acceptance | `test_correction_v366.py` | Implemented |
| Accepted/repair synthesis creates one bounded successor | `855bb11`; transition regression | Implemented |
| Targeted fidelity cannot satisfy design correction | V5 regression | Implemented |
| Result interpretation attacks are scientific, attributed, and deduplicated | `4880485`; correction regression | Implemented |
| Inferred migration labels preserve original history | `081ad41`; correction regression | Implemented |
| V5 missing-attack generation | Read-only bounded proposal preview only | Partial: authorized planner materialization remains required |
| Full legacy V5 A3/A4/A5/A6 extraction | No project-specific historical IDs are hard-coded or fabricated | Partial: audit support exists; historical mapping needs authorized review |
| Scheduler policy for all correction phases | Design successor is materialized; generic scheduler classification is present | Partial: dedicated interpretation-phase scheduling remains to be added |
| Cockpit/UI full three-layer views and audit controls | Basic scientific-design correction panel exists | Partial |
| Confirmation/development separation | Existing v3.6.5 paths unchanged; no correction record grants execution | Preserved |

Latest broad regression: `398 passed, 102 skipped` on Windows. The correction
PostgreSQL suite has `7 passed` as of 2026-08-31.
