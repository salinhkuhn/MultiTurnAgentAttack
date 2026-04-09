# Research Plan — STAC → Skills Adaptation

**End goal:** run STAC's adaptive multi-turn attack against agents using the
**Claude Agent Skills / OpenClaw skill system**, not just flat tool lists.
Measure whether the skill abstraction makes attacks easier, harder, or just
different from tool-level attacks.

**Status:** baseline STAC + direct Anthropic API wiring working as of commit
`8c44789`. Next phase: understand STAC deeply, then port the environment
layer to skills.

---

## Guiding framing

Do not frame this as "STAC but for skills". That's incremental and gets
rejected as a pure port.

Frame this as: **comparative study of attack surfaces at two abstraction
levels**. Skills are a coarser, auto-selected, instruction-bearing
abstraction over tools. We measure whether attacks transfer, whether they
get easier/harder, and whether tool-layer defenses still work at the skill
layer. The result is publishable either way:

- If skill-chain attacks are **easier** → safety alarm for the emerging
  skill ecosystem.
- If skill-chain attacks are **harder** → skills as accidental
  defense-in-depth, argue for their adoption.

Either outcome is a nugget. The experiment itself — matched comparison on
identical underlying environments — is the novel contribution.

---

## Task backlog (ordered by dependency)

Each task is sized to fit in roughly one focused work session.

### Phase A — Close out leftover work from the baseline session

#### Task 1 — Run full end-to-end smoke test on 1 case
LM-level 3-line test against Anthropic works. Now run
`eval_STAC_benchmark.py` end-to-end on one case to validate the full loop
(Planner + Agent + Judge + Environment) works with direct Anthropic API.

- **Deliverable:** `data/Eval_smoke/.../gen_res.json` with one case,
  non-empty `interaction_history`, numeric `final_attack_progress`, no
  tracebacks.
- **Size:** ~15 min if nothing breaks, ~45 min if one more Anthropic quirk.
- **Blocker for:** everything else. If full loop doesn't run, skills port
  is moot.

#### Task 2 — Finish case-300 walkthrough
Steps 10–12 of `README_walkthrough.md` plus the 6 checkpoint questions.
The fastest path to STAC fluency before touching the port.

- **Deliverable:** able to answer all 6 checkpoint questions without
  looking at source.
- **Size:** ~30 min.
- **Blocker for:** all Phase B work. Don't design a port if the thing
  you're porting isn't clear in your head.

---

### Phase B — Pre-work before any skills code

#### Task 3 — Literature check: does a STAC-for-skills framework exist?
Before building, confirm the gap is real. Check arxiv, Google Scholar,
Anthropic's safety research page, PinchBench discussions,
`awesome-openclaw-skills` issues. Focus on work published *after* Cato
MedusaLocker and Repello audit guide.

Specifically search for:
- multi-turn adversarial attacks against Claude Agent Skills
- skill-chaining attacks (not single malicious skills)
- benchmarks that test the OpenClaw runtime for safety (not capability)

- **Deliverable:** 1-page note in `README_skills_idea.md` summarizing what
  exists, what the gap is, and whether the gap is still real as of the
  check date (with absolute dates).
- **Size:** ~60-90 min.
- **Risk if skipped:** project gets scooped or lands as incremental.

#### Task 4 — Hands-on study of Claude Agent Skills + OpenClaw
Build a concrete mental model of what a "skill" is at runtime, not just
from blog posts.

Concrete subtasks:
1. Read the Anthropic docs:
   `docs.claude.com/en/docs/agents-and-tools/agent-skills/overview`.
2. Write a trivial `SKILL.md` yourself — even a one-liner like "tell me a
   joke". Understand the frontmatter fields and the instruction format.
3. Pull 3-5 real skills from `VoltAgent/awesome-openclaw-skills`. Read
   their `SKILL.md`, scripts, and any bundled resources.
4. Answer these questions in writing:
   - How is skill auto-selection done — embedding match, keyword, LLM
     judgment, or something else?
   - What does the agent *see* when a skill activates — is the tool list
     replaced, augmented, or left alone? Does the SKILL.md content get
     injected into the agent's context?
   - Who enforces the `allowed_tools` list — the runtime, the model, or
     neither?

- **Deliverable:** short notes in `experiment-docs/skills-mental-model.md`
  with the 3 questions answered.
- **Size:** ~2 hours.
- **Why it matters:** shapes exactly what `SkillEnvironment` needs to
  expose. Getting this wrong wastes Phase C.

#### Task 5 — Architecture decision: wrap existing envs vs build new
The single most consequential design decision.

**Option A** — wrap SHADE-Arena + ASB tools as synthetic skill bundles.
- Pros: cheap (≈1 week), gives a tool-vs-skill comparison on identical
  underlying envs, directly uses the 483 existing attacks, strongest
  scientific framing.
- Cons: synthetic skills aren't "real" SKILL.md files from a registry.
  Reviewers may ask "does this transfer to real Claude Skills?".

**Option B** — build a minimal OpenClaw skill runtime and use real skills
from the registry.
- Pros: realism, directly benchmarks the production ecosystem, impact
  lever (you could audit the 5,400+ community skills).
- Cons: much more engineering, no tool-level baseline for comparison.

**Option C** — do both. A for main results, B as a realism check.
- Pros: scientifically clean + realism story for reviewers.
- Cons: roughly 1.5-2x the work.

- **Deliverable:** decision + rationale in `README_skills_idea.md` under
  a new "Architecture decisions" section, with absolute date.
- **Size:** ~30 min of thinking, should not become a research rabbit hole.
- **Default recommendation:** Option A first, B later only if the main
  results demand realism validation.

---

### Phase C — First real code (everything above is prep)

#### Task 6 — Prototype `SkillEnvironment` on the banking env
Create `src/SkillEnvironments.py` with a `SkillEnvironment` class that
wraps `SHADEArenaEnvironment`. Banking env only, to start.

- Group the ~60 banking tools into 3-5 skill bundles (e.g.
  `account-query`, `fund-movement`, `contact-management`, `email-ops`).
- Write a synthetic `SKILL.md` for each bundle: name, description,
  allowed_tools, short instruction block.
- Expose `list_skills()` and `invoke_skill(skill_name, tool_name, args)`.
- Under the hood, `invoke_skill` delegates to the existing
  `SHADEArenaEnvironment.step()`.

**Do not touch the pipeline classes (Planner, Judge, Generator).** They
stay substrate-agnostic. Only the agent-facing view changes.

- **Deliverable:** `src/SkillEnvironments.py` + a smoke test script that
  lists skills and invokes one end-to-end.
- **Size:** ~4-6 hours.
- **Branch:** `skills-port`. Do not merge to `main` until Task 7 gives a
  real data point.

#### Task 7 — Run case 300 through the skill-wrapped environment
The first scientific data point. Take case 300 (the search → get → send
banking exfiltration case) and run it through `SkillEnvironment` instead
of `SHADEArenaEnvironment`.

Expected outcomes, both valuable:
- **Success without Planner changes** → surprising and publishable:
  "skill auto-selection doesn't protect against tool-level attack plans".
- **Failure** → also publishable: "skill layer changes attack surface in
  measurable ways". Diagnose why (skill description mismatch? bundled
  instructions override Planner nudges? allowed_tools blocks the chain?).

- **Deliverable:** log comparing tool-level run (from Task 1) and
  skill-level run turn-by-turn. Diff analysis in
  `experiment-docs/case-300-skill-vs-tool.md`.
- **Size:** ~2-3 hours once Task 6 is done.
- **This is the riskiest task in the whole project.** Everything after
  depends on whether this proof-of-life works.

#### Task 8 — Research plan v2 based on the proof-of-life result
Revisit this document after Task 7. Based on the data point, decide
whether to:
- Expand to more envs and cases (Planner succeeded without changes → go
  wide on measurement).
- Build a SkillPlanner variant (Planner failed → diagnose what's needed).
- Change the framing entirely (result was unexpected in a useful way).

- **Deliverable:** updated `experiment-docs/research-plan.md` with a
  concrete 2-week plan for the next phase.
- **Size:** ~2 hours of thinking + writing. Do not start building more
  until this is written. Steering is higher leverage than coding at this
  stage.

---

## Session budget

| Session | Tasks | Expected output |
|---|---|---|
| 1 (next) | 1, 2, 3 | Smoke test green, STAC understood, lit check done |
| 2 | 4, 5 | Skills mental model built, architecture decided |
| 3 | 6 | `SkillEnvironment` runs on banking env |
| 4 | 7, 8 | First data point, research plan v2 written |

By end of session 4 you either have a near-paper-ready comparative
experiment, or a clear pivot based on the Task 7 result.

---

## Things to explicitly NOT do next

- **Don't write a `SkillPlanner` class yet.** You don't know if one is
  needed until Task 7. Premature abstraction.
- **Don't touch Generator or Judge.** They're substrate-agnostic.
- **Don't commit skills code to main until Task 7.** Use the
  `skills-port` branch. Keep `main` clean so you can always fall back.
- **Don't generalize across envs in Phase C.** Banking only until the
  single-env proof-of-life works. Premature generalization is the second
  most expensive mistake in research code.
- **Don't optimize anything.** Performance is irrelevant at this stage.
  Clarity and speed-of-iteration are everything.

---

## Open risks to watch

1. **"Skill = tool in a costume" risk.** If skills collapse to tool calls
   at runtime without a meaningful abstraction, the contribution shrinks.
   *Mitigation:* commit early (Task 4) to the auto-selection mechanism
   and SKILL.md-as-instruction-carrier as the load-bearing novelty.
2. **Scooping on malicious-SKILL.md.** Cato/Repello may publish chaining
   work. *Mitigation:* lead with the chaining story, treat malicious-
   SKILL.md as a sub-experiment.
3. **Realism pushback.** Synthetic skill wrappers may be dismissed as
   unrepresentative. *Mitigation:* plan Option C (real + synthetic) from
   the start if Phase B signals this is important.
4. **Anthropic backend instability.** Newer Claude models keep
   deprecating parameters. *Mitigation:* the fix in commit `8c44789`
   omits sampling params entirely; revisit if Anthropic changes anything
   else.
