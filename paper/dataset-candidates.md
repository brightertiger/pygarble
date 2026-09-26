# Dataset candidates (historical shortlist)

Superseded for the executed study: the author selected the published GitHub
collection only, with no new human label audit. See [the frozen protocol](study/protocol.md)
and [results](study/results/report.md). The unexecuted proposals below are
retained as background, not current requirements.

Metadata reviewed on 26 September 2026. This is a source shortlist, not an
acquired or validated benchmark. Dataset contents, versions and hashes still
need inspection. No experiments or label audits have been completed.

The author prefers existing public datasets, including Kaggle, within a
US$100 total budget. Hosting on Kaggle or Hugging Face does not by itself
establish reuse rights, label quality or suitability for this research.

## First sources to inspect

| Source | Proposed role | Evidence and remaining checks |
| --- | --- | --- |
| [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext), initially `wikitext-2-raw-v1` | Candidate legitimate English prose and parent texts for controlled corruption | The card describes Wikipedia articles and lists CC BY-SA 3.0/GFDL. Preserve attribution and check applicable redistribution terms. Audit headings, markup and fragments before assigning clean labels; group by article. This is a language-modeling corpus, not an annotated gibberish benchmark. |
| [Gaskell and Bowern's human-produced gibberish](https://github.com/danielgaskell/voynich), `data/gibberish_transcriptions.zip` | External stress set for human-produced meaningless text | The study recruited 42 volunteers. The repository supplies anonymized transcriptions under a modified MIT license requiring notice preservation and citation. That grant explicitly excludes its meaningful-text and Voynichese collections. Inspect task fit and document/participant grouping; do not label Voynichese as established gibberish. |

These are candidates for complementary analyses, not a ready-made binary
benchmark. Pooling all Wikipedia text as negative and all volunteer text as
positive would confound source with class. Keep the human-produced set as
an explicitly limited external stress test unless a defensible matched
comparison can be constructed. Seek another licensed prose domain before
claiming general cross-domain performance.

Controlled corruptions can supply reproducible paired examples, retaining
parent IDs and seeds. Report these as synthetic stress tests separately from
human-produced gibberish and naturally observed text damage. They do not
establish effectiveness on all real-world corruption.

## Other candidates and reasons to defer

| Source | Current finding | Decision |
| --- | --- | --- |
| [Kaggle: johnwdata/gibberish-text-classification](https://www.kaggle.com/datasets/johnwdata/gibberish-text-classification) | Kaggle's public dataset API returned `licenseName: Unknown` when checked; upstream provenance and label definitions remain unresolved. | Hold until permissions, origin and labels are verified. A starter notebook's license does not license the dataset. |
| [Kaggle: circuitovertime/gibberish-vs-meaningful-prompt-with-labels](https://www.kaggle.com/datasets/circuitovertime/gibberish-vs-meaningful-prompt-with-labels) | Public API likewise returned `Unknown`; prompt-domain coverage needs inspection. | Hold pending license, provenance and task-fit checks. |
| [agentlans/garbled-text](https://huggingface.co/datasets/agentlans/garbled-text) | Card declares CC BY 4.0 and describes sentence shuffling, noun/verb permutation and sentence replacement using upstream text datasets. | Possible exploratory coherence stress test after checking upstream terms. Its labels do not directly establish character corruption; exclude from the primary label pool for now. |
| [finiteautomata/gibberish-detection](https://huggingface.co/datasets/finiteautomata/gibberish-detection) | Access is gated behind agreement and contact-information sharing. Contents and full terms were not inspected. | Defer; prefer sources reproducible without gated access. No agreement has been accepted. |

Kaggle metadata can be checked through its
[public dataset search API](https://www.kaggle.com/api/v1/datasets/list?search=gibberish).
An unknown license is an unresolved permission question, not proof that reuse
is prohibited. Record the eventual license text and upstream source rather
than relying solely on a search-result field.

## Admission and pilot procedure

1. Inspect the exact source files and license notices. Pin upstream revisions,
   file hashes and acquisition dates. Keep data terms separate from pygarble's
   code license; publish retrieval recipes where copying is inappropriate.
2. Record original labels and annotation/generation methods. Map them to the
   [study definitions](publication-plan.md#research-question-and-scope),
   keeping uncertain and out-of-scope examples separate.
3. Reconstruct document/parent groups; check exact and near duplicates,
   benchmark lineage and overlap with existing development data. Existing
   upstream train/test names do not guarantee independence for this study.
4. Build a development-only pilot of approximately 600–1,000 examples across
   suitable source groups and corruption types. Keep external stress sets
   separate. If eligible data cannot support this, revise the scope.
5. Prepare a blind, stratified 100-example audit for human review, with the
   original context and rubric available. Include difficult legitimate text
   and ambiguous mappings. Report who reviewed it and any changes made.
6. Freeze final data selection and evaluation choices only after the pilot.
   Keep pilot groups out of the final test and choose final size based on
   statistical precision. Do not reuse a tuned-on test set as fresh evidence.

Existing labels reduce manual effort; they do not remove the author's need
to review task definitions, audit findings and scientific claims. No paid
annotation service is assumed. Dataset preparation and baseline experiments
use local CPU resources and do not require LLM calls.
