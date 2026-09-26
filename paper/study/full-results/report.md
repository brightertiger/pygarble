# Full published-corpus comparison

All released labelled texts are covered; no new human label audit.
English is primary. Other languages are a separate scope diagnostic.
A document is flagged when at least half of its chunks are flagged.

| Method | Positive documents detected | English documents falsely flagged | Mean positive chunk fraction | Mean English chunk FPR |
| --- | --- | --- | --- | --- |
| english | 13/38 | 0/4 | 34.2% | 1.21% |
| english_extended | 20/38 | 0/4 | 47.9% | 1.89% |
| legacy | 13/38 | 0/4 | 32.1% | 0.25% |
| word_lookup | 38/38 | 0/4 | 100.0% | 0.00% |
| entropy_based | 0/38 | 0/4 | 0.0% | 0.00% |
| hf_all | 38/38 | 1/4 | 100.0% | 48.47% |
| hf_strict | 38/38 | 0/4 | 99.6% | 0.14% |

Rates average documents equally; micro counts are saved separately.
Do not treat chunks as independent observations or the corpus's class
balance as deployment prevalence. HF training overlap is unknown.
Both HF policies use the same model: all non-clean winners (primary),
or noise/word-salad winners only (secondary).
