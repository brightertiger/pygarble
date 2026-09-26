# Publication preparation

Status: working draft, not ready for submission. No journal submission or
package release has been made. Requirements checked on 26 September 2026.

The current route and proposed research protocol are in
[publication-plan.md](publication-plan.md): pilot, independent evaluation,
reference application, manuscript, arXiv, journal submission and discovery.
The manuscript below remains a JOSS-format working draft; adapt it only once
the venue and evidence are settled. The author has a verified arXiv account.
The budget is capped at US$100, with a US$0 incremental spending target.
Start source selection with the [dataset shortlist](dataset-candidates.md);
its candidates have not yet been admitted to an evaluation benchmark.

Start with [paper.md](paper.md) and its [bibliography](paper.bib). Visible
`TODO` markers identify facts requiring author input. The author has confirmed
the name Ujjwal Singh Rao and supplied a
[Google Scholar profile](https://scholar.google.com/citations?user=tf4MVAgAAAAJ&hl=en).
Affiliation is confirmed as Independent Researcher, India. No ORCID has been
provided; it is omitted rather than guessed.

## Readiness assessment

| Area | Evidence and remaining work |
| --- | --- |
| Research use | Trident (arXiv:2605.00297v2, §5.3) cites this repository as a prospective postprocessing tool, but explicitly says no postprocessing was applied. This supports relevance, not demonstrated adoption. Actual research-use evidence remains needed. |
| Development history | Repository created 14 September 2025. Git history includes work in September/December 2025 and January/February/July/September 2026. Creation and commit dates do not prove continuous public visibility; confirm that history. Much of the screening work arrived on 26 September 2026, so distinguish it from the older gibberish functionality. |
| Open development | Public repository, MIT code license, changelog, PRs, CI, contributor and security guides exist. Confirm who contributed and how actual use informed changes. Do not treat automation accounts as paper authors. |
| Reproducibility | Existing tests, golden corpora, challenge data and timing scripts provide engineering evidence. Add a research-specific, licensed dataset and reproducible workflow before making research-performance claims. |
| Release | Latest GitHub release inspected is v0.8.0; source contains newer APIs. Choose a reviewed version and an archival DOI later. Package publication is deferred. |
| Metadata | Ujjwal Singh Rao, Independent Researcher, India confirmed. Funding, acknowledgements and contribution details remain to be completed. No DOI has been invented. |
| AI disclosure | Author confirms Codex and Claude. Draft records known Codex assistance; model versions and Claude's scope still need completion. Authors must personally verify outputs and design decisions. |

JOSS screens for more than six months of public, iterative development,
actual research use and sound open-source practices. Its current AI policy
requires disclosure; author/editor/reviewer conversations must be written
by humans, except for translation assistance. See the
[submission requirements](https://joss.readthedocs.io/en/latest/submitting.html).
Repository age alone does not establish eligibility.

## Research evidence to develop

The author is not currently sure of an implemented research use case. Keep
that question open rather than presenting the Trident citation as adoption.
The strongest documented connection concerns gibberish detection; the
new PII, secret and profanity modules need their own application evidence
if they are central to the paper's research contribution.

The proposed primary study concerns English corpus-quality assessment,
comparing local detectors with independent data and explicit error costs.
A filename study motivated by Trident remains optional future work. Neither
study has been completed; see the publication plan for the protocol and
decision points. Do not infer malware-detection improvement from text
classification results.

## Work to complete

1. Supply the research context: research question, users, data, the role of
   pygarble, version/configuration, and a link or other verifiable evidence.
2. Refine the research contribution and compare relevant alternatives on
   that same use case. The bibliography contains starting points, not a
   completed literature review. Do not claim accuracy or speed superiority
   without a comparable evaluation.
3. Record dataset provenance, licensing, annotation protocol and train/test
   separation. Preserve difficult clean negatives, especially domain terms
   and multilingual text. Report false positives as well as detections.
4. Measure native and optional backends separately. Record hardware, Python
   and backend versions, input sizes, cold/warm timings and subprocess costs.
   Existing synthetic timing results are not a research benchmark.
5. Complete metadata and all `TODO` sections; have every author review the
   technical claims, citations, AI disclosure and authorship.
6. Render the final manuscript with JOSS tooling, inspect it, and complete
   the [review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html).
   Prepare the reviewed release/archive separately when ready.

## Preview

The current [paper format](https://joss.readthedocs.io/en/latest/paper.html)
uses YAML metadata, Markdown and BibTeX, with a 750–1750-word body. A generic
HTML preview checks citation parsing but is not the journal's PDF proof:

```bash
cd paper
pandoc paper.md --citeproc --standalone --to html -o /tmp/pygarble-paper.html
```

Once metadata is complete, JOSS documents this local PDF build (requires
Docker; run from the repository root):

```bash
docker run --rm \
  --volume "$PWD/paper:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

Keep generated previews outside commits. A successful render does not mean
the scientific claims or submission eligibility have been validated.
