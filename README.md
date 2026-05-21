# SOLVED - CONCLUSIONS
High certainity is not an achievement if model memorizes instead of learning

Problem:
AI models are not able to analyze their own knowledge efficiently enough to reliably mark an answer as unknow/uncertain.

Hypothesis:
Constraining the number of available neuron paths and analyzing activation variance on simple tasks may allow uncertainty to emerge as a measurable property of the system.

Motivation:
Implementation of reliable uncertainity scale in AI models carries various benefits:
  - Fewer halucinations;
  - Model's ability to request specific data examples it needs to learn:
    > Decent accuracy with little training data;
    > Possibly suggesting new dependencies and experiments for science questions.

Initial scope:
  - numerical sequences;
  - simple symbolic dependencies;
  - low-parameter neural architectures;
  - uncertainty scoring based on internal variance;
