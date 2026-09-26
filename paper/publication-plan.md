# Publication plan and current status

Updated 27 September 2026 after the author requested autonomous execution
using the published GitHub dataset without a new human audit. This replaces
the earlier proposal for a larger pilot and additional annotation. Historical
plans remain in Git history.

The technical study is complete for review. Start with
[the full results](study/full-results/report.md), [the empirical draft](study/manuscript.pdf)
and [the full protocol](study/full_corpus_protocol.md). All research code and artifacts
are together in `paper/study/`. No submission or merge has been made.

## Completed research

The full study compares unchanged pygarble configurations with a pinned local
Hugging Face DistilBERT classifier across every released labelled text in the
Gaskell/Bowern GitHub corpus. All 109 documents and 31,964,664 normalized
characters are covered through 79,969 chunks. English is the primary target;
the 67 other-language meaningful documents are a separate scope diagnostic.
Two HF label mappings are fixed in advance and both are reported. CPU latency,
process memory, document aggregation and tail sensitivity accompany accuracy
measurements. Source labels are inherited without a new human audit.

The original capped comparison and character-model calibration results remain
available as preliminary evidence. The extension overlaps those data and is
not an independent confirmation. More text does not remove the limited source
diversity: only four meaningful English documents and 38 positive transcripts
support the primary comparison. The transformer training-data overlap is
unknown. Full coverage, fixed subset neural replay and the preliminary
byte-identical reproduction are distinct checks with explicit scope.

This is a completed author-requested, AI-assisted analysis awaiting scientific
review, not independent external adoption. PII, secret and profanity screening
are outside its evaluation scope. The Trident reference remains prospective
use, not evidence that pygarble contributed to that paper's results.

## Route within the budget

The author has a verified arXiv account. The proposed route is a reviewed
empirical preprint, followed by consideration of a distinct short software
paper for fee-free JOSS. The JOSS article should explain the software and
research use rather than repeat the empirical results article. Disclose the
related manuscript. JOSS's scope and development-history criteria still apply.

The total spending ceiling is US$100. No paid data, inference, annotation or
compute service was procured. The incremental paid-service target remains
US$0, using existing hardware and free distribution. This excludes existing
equipment, subscriptions and author time. JORS's listed GBP824 software-paper
fee exceeds the budget; it is an alternative only with a confirmed waiver
that keeps the total within the ceiling.

## Next steps after the author returns

1. Review methods, results, scope and scientific claims. Dataset labels remain
   inherited; this step is manuscript/scientific review, not new annotation.
2. Complete author declarations and AI disclosure. Confirm authorship,
   affiliation, funding/conflicts, Claude's scope and model versions where known.
3. Finalize the empirical manuscript and choose an arXiv category/license.
   The author submits the reviewed source bundle; account verification does
   not remove moderation or category-specific requirements.
4. Assess JOSS readiness against the current public-development and substantial
   research-software criteria. An executed analysis helps demonstrate use but
   does not ensure significance or admission to review. Finalize a reviewed
   software version and archival DOI if proceeding; no DOI is invented now.
5. After public release, add accurate citation metadata and links to the
   arXiv/journal record, update the Scholar profile and check exact-title search.
   A profile entry or Search Console verification does not confirm Scholar
   indexing. Indexing can take time and is not guaranteed.

The detailed [submission handoff](study/submission.md) separates completed
technical preparation from the final author review and submission.

## Official guidance checked

- [JOSS submission, scope and AI policies](https://joss.readthedocs.io/en/latest/submitting.html)
- [arXiv submission overview](https://info.arxiv.org/help/submit/index.html)
- [Google Scholar inclusion](https://scholar.google.com/intl/en/scholar/inclusion.html)
- [JORS requirements and fees](https://openresearchsoftware.metajnl.com/about/submissions)
