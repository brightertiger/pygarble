# Author review and submission handoff

Status: the author requested arXiv submission, then requested revisions to the
title, software emphasis, layout and metrics section. The revised paper is under
author review. An unfinished draft was opened in the logged-in arXiv workflow;
no files have been uploaded and no announcement submission has been completed.
JOSS has not received a submission. No new dataset audit is planned.

## Review packet

- `manuscript.pdf`: empirical paper preview.
- `manuscript.md` / `manuscript.template.md`: rendered text and editable source.
- `manuscript.tex`: compilable TeX with resolved references.
- `review-source.tar.gz`: TeX, bibliography and the referenced figures;
  an author-review source bundle, not a declaration that it is ready to submit.
- `full-results/report.md`: complete corpus results; JSON/CSV files hold outcomes.
- `results/report.md`: retained preliminary capped comparison and calibration.
- `full_corpus_protocol.md`, `sources.json`, `hf_model.json`, `LICENSES.md`,
  `validation.md`: provenance,
  scope, rights, tests and reproduction evidence.
- `../paper.md`: separate JOSS-format software article draft.

The data labels come from the published collection. The author still needs
to review the research design, interpretation and AI-assisted text before
claiming responsibility for a paper. Automated checks are not that review.

## arXiv preparation

Title: **pygarble: Modular, Low-Cost Gibberish Detection and Text Screening in Python**.

Author byline: **Ujjwal Singh Rao** only, as requested. The previously supplied
background affiliation is Independent Researcher, India; it is omitted from
the paper byline. Proposed category: `cs.CL`, subject to fit and endorsement.
Use the abstract in the final reviewed manuscript. No ORCID, public arXiv
identifier, journal reference or DOI has been invented.

The author confirmed no funding and no conflicts of interest. The paper also
identifies the author as pygarble's developer. Claude assisted with coding and
running comparison experiments; Codex's broader study/manuscript assistance
is disclosed. Exact model versions were not recorded. The author selected
**arXiv's perpetual, non-exclusive distribution license**.

The source bundle has a conventional two-column article body, numbered
sections/equations/captions and full-width supplementary tables. The revised
paper has not been accepted or reviewed by a journal. Before completing the
arXiv workflow, review this revised scientific content and the actual account
submission agreement, choose the confirmed license, upload the source bundle
and inspect arXiv's own compilation preview. Local compilation is not a test
of arXiv's TeX environment. Use XeLaTeX for the locally tested Unicode build.

No agreement or author attestation has been checked in the browser workflow.
The account's displayed affiliation differs from the previously supplied
paper affiliation; verify the contact declaration before certifying it.
The workflow remains at its start page while manuscript revisions are reviewed.
See the [official submission overview](https://info.arxiv.org/help/submit/index.html).

## JOSS decision

Use the short software draft in `../paper.md`, not this results-focused
empirical manuscript. Disclose the related preprint and link the completed
workflow. The original Trident mention remains prospective and must not be
presented as an implemented use of pygarble.

Before submitting, assess substantial contribution, actual research utility,
public development over time, packaging and archival release readiness.
This small automated study adds executed research evidence but does not prove
independent adoption or guarantee JOSS scope acceptance. Review the
[current requirements](https://joss.readthedocs.io/en/latest/submitting.html).
The author must personally handle editor/reviewer conversations under JOSS's
current AI policy; AI-authored exchanges are not included in this packet.

No journal fee is planned. JORS is only a fallback if a confirmed waiver keeps
all spending within the US$100 total ceiling. No paid service was procured.

## After public release

Link the real arXiv record from the repository and an article landing page;
use consistent title/author metadata and an accessible abstract/PDF. Add the
journal reference later if accepted. Search Google Scholar by exact title
after publication and distinguish a manual profile addition from globally
indexed search results. Do not claim indexing until it is observed. See
[Scholar's inclusion guidance](https://scholar.google.com/intl/en/scholar/inclusion.html).
