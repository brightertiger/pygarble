# Protocol deviations and corrections

The initial broad publication plan was narrowed before scoring to the
published GitHub corpus at the author's request. No new human audit, Kaggle
data, synthetic labels or 600–1,000-example pilot is part of this study.
The released corpus's size and source structure motivated the protocol in
`protocol.md`. Further deviations must be recorded here with their timing.

After scoring, only reporting and review tooling was added: plots, manuscript
rendering, source packaging and saved-output verification. No evaluation code,
threshold, source membership or result was changed. PDF pagination was adjusted
to keep a table within the page; this changes presentation only.

## Full-corpus extension

On 27 September 2026 the author requested every released labelled GitHub
text and a Hugging Face benchmark. `full_corpus_protocol.md` and new core
scoring modules were frozen in commit `717ceee` before the new predictions.
The original capped outcomes were already known. This is an exploratory
extension, not an independent confirmation or a retroactive change to the
original protocol. Original evaluation source files and results are retained.

The extension includes all 109 released labelled documents and all normalized
characters through 79,969 chunks, with both HF policies fixed in advance.
No model, package default, source membership or threshold was tuned after
viewing results. Reporting, figures, runtime measurement and artifact/replay
verification were added during the long full-corpus inference run. Supplementary
applicability and pairwise disagreement tables summarize saved predictions;
they add no trained model or selected operating point.

Final reporting checks clarified that word lookup is not a member of the
three tested ensemble profiles and that zero scores on text without eligible
Latin-letter words do not establish linguistic coverage. The paper describes
these implementation details explicitly. The installed tokenizer version was
added to the optional requirements for reproduction; no installed version or
inference setting changed. PDF-only pagination changes resolved overfull boxes.

## Additional chunk summaries and manuscript revision

After inspecting the full results, the author requested accuracy and confusion
matrices. `chunk_metrics.py` derives these post hoc descriptive measures from
the saved target-comparison predictions and adds an always-keep reference.
The reporting tests independently recount the same predictions. No corpus
membership, inherited label, model, package configuration, threshold or stored
prediction changed. The frozen source-level endpoints, intervals and sensitivity
analyses remain in Appendix A; they were not replaced retroactively in the
protocol.

The manuscript was subsequently reorganized around pygarble's software
architecture and strategies, followed by the empirical comparison. Two-column
article typesetting, numbered equations/captions and a name-only byline are
presentation changes. Funding and conflict declarations and Claude's scope
were supplied by the author. Screening capabilities are described without
claiming they were evaluated by the gibberish study.
