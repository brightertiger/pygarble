# pygarble paper and benchmark

The [paper PDF](study/manuscript.pdf) describes pygarble's architecture and
strategies, then evaluates gibberish detection against local DistilBERT.
[Study instructions](study/README.md), [full results](study/full-results/report.md)
and [validation](study/validation.md) explain the data, reproduction and limits.
The paper study is the reference for reported gibberish benchmark results;
`paper/regression/` retains fast engineering checks for the broader library.

## Contents

- `study/manuscript.template.md`, `references.bib`, `article.tex`, `tables.lua`
  and the two referenced figure PDFs: editable paper sources and layout.
- `study/manuscript.pdf`: review copy, retained for easy access.
- `study/`: pinned protocols, provenance, study code and tests.
- `regression/`: engineering evaluation scripts, fixtures and golden outputs.
- `scripts/`: package data generation, strategy docs and discovery checks.
- `study/full-results/`: frozen complete-corpus measurements and predictions.
- `study/results/`: preliminary results referenced by the paper; these overlap
  the full corpus and are not independent replication.
- `paper.md`, `paper.bib`: separate short JOSS-format software draft.

Generated Markdown, TeX and the upload archive live in ignored
`study/.cache/publication/`. Rebuild from the repository root with `make paper`
(Pandoc and Tectonic required). This updates the review PDF and creates
`study/.cache/publication/review-source.tar.gz`. The bundle contains compilable
TeX, bibliography and both figures, not raw corpus text or model weights.
Obsolete planning documents and duplicate previews are retained in Git history.
Historical paths mentioned by the frozen protocols refer to that earlier
revision; the protocols themselves remain unchanged for provenance.

## Publication status

Author: **Ujjwal Singh Rao**, with a name-only paper byline. The author confirmed
no funding, no conflicts, Codex assistance and Claude's coding/experiment
assistance. The author is also the software's developer. No ORCID was supplied.

The author authorized arXiv submission and selected the perpetual,
non-exclusive distribution license. The Chrome workflow is blocked before
upload because the account needs **cs.CL endorsement**. arXiv sent the request
to the author's email. No paper has been submitted for announcement or accepted;
a public identifier must not be invented. Once endorsement is granted, resume
the existing draft, upload the rebuilt bundle, and check arXiv's compiled PDF
and final metadata. Local compilation does not verify arXiv's TeX environment.
See [arXiv's endorsement instructions](https://info.arxiv.org/help/endorsement.html).

The JOSS draft is separate and still requires assessment against the journal's
software contribution, development history, research use and archival-release
requirements. The completed author-led evaluation is not independent adoption.
Disclose the related manuscript; do not submit the longer empirical article as
the short JOSS paper. Official Inara rendering has not run locally because
Docker is unavailable; the generic Pandoc citation preview has passed.

No paid service has been procured; the external-spend ceiling remains US$100.
Package publication remains deferred. After an actual public release, add the
real paper links/citation metadata and check Scholar indexing; neither journal
acceptance nor indexing is guaranteed.
