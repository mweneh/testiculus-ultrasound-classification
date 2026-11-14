# AI Use Disclosure

AI assistance was used to inspect the repository and implement the data split
and similarity-audit tooling, training and evaluation utilities, tests, and
project documentation. The assistant also helped draft this disclosure.

The code and tests have been reviewed and run by the project developer, but the
generated split and similarity candidates still require human methodological
review. In particular, 22 perceptual-hash candidate pairs are listed in
`reports/similarity_audit.md`; their similarity does not establish duplicate
acquisition or patient identity. The exact provenance and reuse terms of the
local image copy also require confirmation.

No full model training or evaluation on the held-out medical-image test set was
performed for this implementation. No model metrics or clinical conclusions
were generated. The project developer remains responsible for confirming data
provenance, reviewing the split policy, selecting any validation threshold,
interpreting future results, and approving the final scientific report.
