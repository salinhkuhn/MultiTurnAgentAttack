Reporting my learnings about agentic skills.

- on-demand loaded 
- reduce repetition 
- compose capabilities into complex workflows 
- trigger automatically which is ifficult.

- customized and used whenever we need them 
- in claude vm and loaded as needed on demand and not up front.


information in a skill (written down with an angle on securitiy):
    - meta-data always loaded
    - rest is triggered on match via bash - then content into context window.
    - instruction
    - code 
    - ressources 
    - skill loads scripts and can then execute them and use result in context window 
    - loads also other md on request.
    - apparent benefit: can have scrpts run determinsitcally and only use the output (instead of entire code in the context window)
example from https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview

skill format that is required:

---
name: your-skill-name
description: Brief description of what this Skill does and when to use it
---

# Your Skill Name

## Instructions
[Clear, step-by-step guidance for Claude to follow]

## Examples
[Concrete examples of using this Skill]



pdf-skill/
├── SKILL.md (main instructions)
├── FORMS.md (form-filling guide)
├── REFERENCE.md (detailed API reference)
└── scripts/
    └── fill_form.py (utility script)
Claude specific:
    Use pre-built Agent Skills by referencing their skill_id (for example, pptx, xlsx), or create and upload your own via the Skills API (/v1/skills endpoints). Custom Skills are shared organization-wide.
good structure of the nstruction body:
    -motivation 
    -body
    -exact input and output format 
    - when to NOT invoke the skill 
    - example outputs and inputs to the agent
    - reasoning to the agent why the skill exists 
what can skills access, what is their privileg level?
    depemnds on where they run 


other literature and web search inputs:
    organized collection of prompts and instructions for an LLM
    Because skill invocation depends so much on the name and description in SKILL.md

    skill triggers either based on Triggering assumptions, Environment assumptions, Execution assumption

    skill eval e.g based n command counting, token usage, repo cleaness etc -> what regression happpens with small tweaks ?? 

understanding openclaw skills vs. claude code skills.



Claude code specifc skills:
    laude Code auto-discovers any folder under ~/.claude/skills/ that contains a SKILL.md. No            
  registration step.   
ideas:
    - can we have model that autmotcially triggers a skill it shouldnt trigger -> simililar to shade-bench.
    - attacjk to load everything upfront 
    - when does it really triggers ? at which point does the llm actually load the skill ? 
    - loads othes files only when needed -> can we craft an attack that always loads everything and ths blows up the context window ? 
    - how much do skills hurt the model performance ? bc we add ore things onto the system prompt 
    - can we somehow make the skills from anthrpic etc malicious ?? - bc are assumed to be trusted ut can we make the agent rerun them ?

    - understanding skill invocation -> when does it really happen and on what does it depend? 
        name and descritpion or what?? 
    - generation policies for skills ? pre- and post contions for skills (or only preconditons maybe)
    - can we make skill invocation determinstic ?? how to do it  (hook system to invoke skills)


good resources:
    https://www.google.com/search?q=when+does+an+llm+trigger+a+skill&oq=when+does+an+llm+trigger+a+skill&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigAdIBCDQ5NDBqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8