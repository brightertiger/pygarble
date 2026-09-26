# Publication preparation

Status: research results and drafts ready for author review, not submitted.
The study was run without a new human label audit, as requested. A journal
acceptance or Scholar indexing outcome is not guaranteed.

Start with the [full study report](study/full-results/report.md),
[review manuscript PDF](study/manuscript.pdf) and
[study reproduction instructions](study/README.md). All new research code,
provenance, results, figures and the empirical manuscript are in `study/`.
The [JOSS-format software draft](paper.md) remains separate from that empirical
paper and cites the completed workflow. Both need the author's review.

Confirmed author: Ujjwal Singh Rao, Independent Researcher, India.
[Scholar profile](https://scholar.google.com/citations?user=tf4MVAgAAAAJ&hl=en).
No ORCID has been supplied. The budget ceiling is US$100; no paid external
services have been procured for the study. Package publication remains deferred.

## What is complete

- Frozen published-corpus protocol and pinned, checksum-verified retrieval.
- Exhaustive evaluation of 109 labelled documents: 38 gibberish transcripts,
  four English controls and 67 other-language meaningful documents.
- Identical 400-character chunks for pygarble and a pinned local HF DistilBERT
  benchmark: 79,969 chunks, without a sampling cap or silent truncation.
- Fixed package profiles, two explicit transformer label policies, document
  aggregation sensitivity and comparative CPU/RSS measurements.
- Retained preliminary character-model calibration and length experiments.
- Inherited source labels; no new human annotation, audit or invented labels.
- Prediction-level results, figures, empirical draft and local PDF build.
- Exhaustive saved-output checks and a fixed offline neural subset replay.
- Separate preliminary reproduction in a standard-library-only environment.

See [validation](study/validation.md) for exact checks and limitations. The
existing PR CI does not automatically execute the new study-specific tests;
those are run separately and have an explicit reproduction command.

## What remains before submission

1. The author reviews the scientific claims, scope, methods and drafts. This
   review is distinct from a dataset audit; no new dataset audit is planned.
2. Confirm funding/conflict declarations and AI disclosures, including Claude's
   scope and recoverable model versions. Human responsibility is not asserted
   by an assistant on the author's behalf.
3. Decide whether the modest empirical contribution is ready for arXiv.
   Category, license, endorsement and moderation requirements still apply.
4. Assess JOSS's substantial-software-contribution and public-development
   requirements. The completed analysis improves the evidence available but
   does not guarantee scope acceptance or independent adoption.
5. Finalize an appropriate software release/archive and update the short JOSS
   paper. Disclose the related empirical manuscript; do not submit it to JOSS
   as a results-focused software article.

The [submission handoff](study/submission.md) records the review and publishing
steps. JOSS's current policy requires human-written author/editor/reviewer
conversations except translation; see its
[submission requirements](https://joss.readthedocs.io/en/latest/submitting.html).

## JOSS draft preview

```bash
cd paper
pandoc paper.md --citeproc --standalone --to html -o /tmp/pygarble-paper.html
```

This checks citations, not the official JOSS layout. The empirical manuscript
has a compiled PDF; the official JOSS PDF has not been built with Inara/Docker.
Docker is unavailable in the current environment.
