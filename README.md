Problem:
AI models are not able to analyze their own knowledge efficiently enough to reliably mark an answer as unknow/uncertain.

Hypothesis:
By giving a model little amount of neuron paths and tracking variance on simple test cases, we can reliably measure uncertainity.

Motivation:
Implementation of reliable uncertainity scale in AI models carries various benefits:
  - Fewer halucinations;
  - Model's ability to request specific data examples it needs to learn:
    > Decent accuracy with little training data;
    > Possibly suggesting new dependencies and experiments for science questions.
    
