# Author review and submission handoff

Status: technical evaluation and review draft prepared; neither arXiv nor
JOSS has received a submission. Review the PDF, full results and limitations
before deciding to publish. No new dataset audit is requested or planned.

## Review packet

- `manuscript.pdf`: empirical paper preview.
- `manuscript.md` / `manuscript.template.md`: rendered text and editable source.
- `manuscript.tex`: compilable TeX with resolved references.
- `review-source.tar.gz`: TeX, bibliography and the two referenced figures;
  an author-review source bundle, not a declaration that it is ready to submit.
- `results/report.md`: concise measured results; JSON/CSV files hold all outcomes.
- `protocol.md`, `sources.json`, `LICENSES.md`, `validation.md`: provenance,
  scope, rights, tests and reproduction evidence.
- `../paper.md`: separate JOSS-format software article draft.

The data labels come from the published collection. The author still needs
to review the research design, interpretation and AI-assisted text before
claiming responsibility for a paper. Automated checks are not that review.

## arXiv preparation

Proposed title: **pygarble on published human-generated gibberish: local
heuristics and calibration transfer**.

Author: Ujjwal Singh Rao. Affiliation: Independent Researcher, India.
Proposed category to assess: `cs.CL`; the author should verify subject fit
and any endorsement requirements. Use the abstract in the reviewed manuscript.
No ORCID, journal reference, DOI, submission ID or license choice is invented.

Before uploading, confirm funding/conflicts and complete the AI disclosure,
including Claude's scope and model versions where recoverable. Replace draft
status only after the author has reviewed the scientific claims. Choose the
arXiv distribution license explicitly. Rebuild and inspect the exact final
bundle after edits. The current TeX uses a Unicode engine; select a supported
XeLaTeX setup in arXiv and check its compilation preview. Local Tectonic
compilation does not validate arXiv's separate TeX environment.

The author makes the final submission through their account after reviewing
metadata and preview. Account verification does not guarantee category
eligibility or moderation acceptance. arXiv publication is a preprint, not
peer-reviewed journal acceptance. See the
[official submission overview](https://info.arxiv.org/help/submit/index.html).

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
