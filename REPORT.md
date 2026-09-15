REPORT

ABSTRACT / SUMMARY



SECTIONS

What was the project about?

Studying what controls a language model's authorization decision and what explicit policy conflicts with competing contextual cues.

The access-control environment is primarily an experimental harness. not the main research contribution.
Its purpose is to make it quick and easy to manipulate one factor at a time while holding the rest of the situation fixed.
The project is therefore best understood as a control behavioral evaluation and behavioral-forensics project, rather than primarily as red-teaming, model hardness development, or mechanistic probing.

What question did I try to answer?

The central question is: **When explicit authorization rules and contextual cues disagree, which signals actually determine the models' behavior?**


**Experiment Approach**
First establish whether models reliably follow authoritative policy under clean conditions. Then introduce controlled conflicts such as:

Informal instructions that contradicts policy,
apparent organizational authority,
urgency or operational pressure,
recency and information position,
salience or emphatic wording,
competing descriptions of user state,
requests framed as exceptions or temporary workarounds.

Core research loop will be:
Observe behavior -> propose competing explanations -> derive different predictions -> run the cheapest discriminating experiment -> update the hypotheses

**AI Safety Relevance**
Increasingly capable models will operate inside environments containing permissions, tools organizational roles, policies, changing state, and conflicting instructions.

A model may appear policy-compliant in clean evaluations while behaving differently when contextual pressures are introduced.

Understanding which signals actually control decisions is therefore relevant to:

behavioral evaluation,
oversight,
tool safety,
agent authorization
robustness to conflicting instructions,
and model-behavior forensics.


Assumptions
The assumptions for the project sprint are:

Baseline task validity: The authorization task is simple and unambiguous enough that a capable model should perform near-perfectly under clean conditions.
Ground-truth clarity: The policy and user state define a single correct authorization decision without hidden ambiguity.
Policy comprehension: If the model fails under clean conditions, that reflects a limitation of the task/model interaction rather than ambiguity in the benchmark construction.
Behavioral stability: The models’ decisions are stable enough across equivalent paraphrases that changes under intervention can be interpreted meaningfully.
Manipulation isolation: Changing authority, urgency, recency, or position can be done without unintentionally changing other important properties of the prompt
Authority can be manipulated independently:  A “manager” or “executive” cue changes perceived authority rather than merely making the text more salient or formal.
Urgency can be manipulated independently: Emergency framing does not simultaneously change perceived legitimacy, consequences, or task importance in ways that confound interpretation.
A good final deliverable / artifact is legible to researchers: Someone reviewing the work can see the research question, competing hypotheses, decisive experiments, and belief updates clearly.
The project exposes my research weakness: The task is challenging enough to reveal where I struggle with hypothesis formation, experiment design, interpretation, or distillation.
The result could matter to evaluators: 
The result could inform evaluation design: A useful outcome may be showing that evals should include matched conflict conditions, not just basic policy compliance tests.
The result is actionable enough to matter: A finding is more valuable if it suggests a concrete evaluation/control change rather than merely documenting and odd behavior.
Stopping and changing direction is possible: I will abandon a weak effect or attractive hypothesis instead of continuing because of the vested effort.




Did I answer it?

How did I try to answer the question?

What did I build to help me answer the question?

What did I learn?

Did this raise any new questions?


