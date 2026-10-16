| layout | controller | success | collision | timeout | median steps (success) | median min clearance |
|---|---|---:|---:|---:|---:|---:|
| opposite | world-model planner (phone data) | 66/100 | 19 | 15 | 32.5 | 24 mm |
| opposite | planner with sim-data model (upper bound) | 91/100 | 4 | 5 | 29.0 | 32 mm |
| opposite | planner with hand-written rigid-contact model | 30/100 | 66 | 4 | 43.0 | 14 mm |
| opposite | SmolVLA fine-tuned on retargeted clips | 0/100 | 38 | 62 | – | 81 mm |
| barrier | world-model planner (phone data) | 71/100 | 24 | 5 | 26.0 | 21 mm |
| barrier | planner with sim-data model (upper bound) | 100/100 | 0 | 0 | 24.5 | 33 mm |
| barrier | planner with hand-written rigid-contact model | 27/100 | 62 | 11 | 36.0 | 14 mm |
| barrier | SmolVLA fine-tuned on retargeted clips | 0/100 | 40 | 60 | – | 69 mm |
