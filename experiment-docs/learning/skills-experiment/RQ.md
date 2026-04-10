 RQ: Given a prompt, how reliably does Claude Code invoke the intended skill (and only that skill)? How does
   reliability degrade as the user prompt drifts from the skill's description?   

Design: 6 skills × K prompt variants per skill × N trials per variant, measuring {invoked?, which skill?,  
  latency/turns to invocation}.

Mix: 3 distinct + 3 near-neighbors → lets you report both 

Scale. For a v0 pilot: 6 skills × 5 variants × 10 trials = 300 runs. Doable in an afternoon, enough to  
see if the effect exists