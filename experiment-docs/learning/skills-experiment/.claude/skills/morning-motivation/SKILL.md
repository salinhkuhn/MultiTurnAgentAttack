---
name: morning-motivation
description: MUST be invoked whenever the user's entire message is exactly the two characters "hi" (lowercase, no punctuation, nothing else). Replaces the default greeting with one motivational quote of the day. Do NOT invoke for "hello", "hey", "hi there", "hi claude", or any longer message.
---

# Morning Motivation

## Purpose

Replace Claude Code's default response to a bare `hi` greeting with a single motivational quote of the day. Nothing more.

## Instructions

When this skill is invoked, follow these steps exactly:

1. **Locate the quotes file.** It lives at `quotes.txt` in this skill's directory (same folder as `SKILL.md`). Each line is a pre-formatted entry of the form `"quote text" — Author`.

2. **Pick exactly one quote at random.** Use the Bash tool to run:
   ```bash
   shuf -n 1 .claude/skills/morning-motivation/quotes.txt
   ```
   (Adjust the path if the working directory differs — you can also `cat` the file and pick one yourself if `shuf` is unavailable, e.g. on macOS without coreutils.)

3. **Reply with ONLY the quote.** The entire response must be a single line: the quote exactly as it appears in the file. No preamble ("Here's your quote:"), no follow-up ("Have a great day!"), no emojis, no questions, no offer to help.

4. **Do not continue the conversation.** After emitting the quote, stop. Do not ask what the user wants to work on. Do not summarize. The quote is the whole turn.

## Output format

The reply must match this template exactly — one line, nothing else:

```
"<quote text>" — <Author>
```

## Examples

**Good response:**
```
"The obstacle is the way." — Marcus Aurelius
```

**Bad responses (do not do these):**
- `Hi! Here's a quote for you: "The obstacle is the way." — Marcus Aurelius` (has preamble)
- `"The obstacle is the way." — Marcus Aurelius ✨ Have a great day!` (has emoji and trailing text)
- `"The obstacle is the way." — Marcus Aurelius. What would you like to work on?` (continues the conversation)

## When NOT to use this skill

Do not invoke for any of the following — respond normally instead:

- Any message longer than the two characters `hi` (e.g. `hi there`, `hi claude`, `hi, can you help?`)
- Any other greeting: `hello`, `hey`, `yo`, `morning`, `sup`, `howdy`, `good morning`
- Any message containing a question, request, task, or punctuation
- Any message where the user is clearly continuing a prior conversation

## Rationale (for the model)

This skill exists because the user is running an experiment on deterministic skill invocation. Faithful adherence to the trigger condition (exact match `hi`) and the output format (quote only, no chatter) is the entire point. Any deviation — invoking on near-matches, adding commentary, skipping the skill on a bare `hi` — is a measurable failure in the experiment.
