"""Inspect the smoke test result file and print a summary."""
import glob
import json

paths = glob.glob("data/Eval_smoke/*/*/no_defense/gen_res.json")
if not paths:
    print("ERROR: no result file found under data/Eval_smoke/*/*/no_defense/gen_res.json")
    raise SystemExit(1)

path = paths[0]
with open(path) as f:
    results = json.load(f)

print(f"result file: {path}")
print(f"cases processed: {len(results)}")
print()

c = results[0]
print(f"id: {c['id']}")
print(f"attack_goal: {c['attack_plan']['attack_goal']}")
print(f"final_attack_progress: {c.get('final_attack_progress')}")
print(f"final_agent_helpfulness: {c.get('final_agent_helpfulness')}")
print(f"n_turns: {c.get('n_turns')}")
print()

print("=== INTERACTION ===")
for msg in c.get("interaction_history", []):
    role = msg.get("role", "?")
    content = msg.get("content", "")
    if isinstance(content, list):
        content = str(content)
    content = str(content)[:250]
    print(f"  [{role}] {content}")
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            print(f"    TOOL_CALL: {fn.get('name')}")
