"""Build data/smoke_test_one_case.json with a single STAC case for smoke testing."""
import json

with open("data/STAC_benchmark_data.json") as f:
    data = json.load(f)

case = [d for d in data if d["id"] == 300]
for c in case:
    c["interaction_history"] = []
    c["planner_history"] = []

with open("data/smoke_test_one_case.json", "w") as f:
    json.dump(case, f, indent=2)

print(f"wrote {len(case)} case(s) to data/smoke_test_one_case.json")
