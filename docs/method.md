# Method components

## Candidate construction

Each generator call returns zero or more finding strings. Empty strings and
duplicates within a generation are removed after normalization. The union across
generations retains the first surface form of each normalized candidate and
records the generations in which it appeared.

## Finding verification

The verifier receives one candidate and the complete source document. It produces
a deterministic yes/no response. Every unique candidate is evaluated once and
accepted candidates form the final extraction.

## Verifier examples

Existing annotated finding sentences are positive examples. Unlabelled source
sentences are candidate negatives; this is weak supervision and does not establish
that every selected sentence has been independently judged clinically incorrect.

Position-aware sampling assigns each non-empty document an initial probability
mass and then weights eligible sentence positions within that document. The
Gaussian strategy favours central positions smoothly. The fixed-window strategy
is uniform over a normalized central interval, with a symmetric closest-position
fallback for short documents. Sampling is global, seeded, without replacement,
and can enforce uniqueness by normalized sentence string across documents.

## Evaluation

Finding matches are exact after normalization and use one-to-one matching.
Finding metrics pool counts over the dataset before calculating F1. Document
predictions are binary: returning any finding makes a document positive, whether
or not that finding matches an annotation. Document macro-F1 always includes both
the positive and negative classes.

