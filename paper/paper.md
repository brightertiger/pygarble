---
title: 'pygarble: Local rule-based text screening and English gibberish detection'
tags:
  - Python
  - text processing
  - data quality
  - privacy
authors:
  - name: Ujjwal Singh Rao
    affiliation: 1
affiliations:
  - name: 'Independent Researcher, India'
    index: 1
date: 27 September 2026
bibliography: paper.bib
---

> Working draft. A reproducible evaluation is available for author review;
> disclosure and submission-readiness review remain incomplete. This manuscript has not been submitted to JOSS.

# Summary

pygarble is a Python library for screening text locally before further
processing. It identifies selected patterns associated with personal
identifiers, exposed credentials, English profanity and garbled English
text. Applications can inspect findings or redact their text spans. The
base installation has no runtime dependencies, and built-in detection does
not call a language model or send text to an external inference service.
Optional integrations extend coverage using locally installed tools.

The package separates sensitive-content screening from gibberish detection.
This allows applications to select the checks relevant to their data instead
of treating every unusual string as a content problem. Results identify the
rules and locations responsible for a finding. The library is intended as an
inexpensive initial filter whose outputs can be inspected and followed by
additional analysis. It does not establish that unflagged text is safe,
anonymous, meaningful or appropriate for a particular research dataset.

# Statement of need

Text-processing workflows can require several distinct decisions: whether
an input appears corrupted, whether it contains a structured identifier,
whether a credential may have been exposed, and whether selected vocabulary
should be flagged. These decisions have different error costs. Removing
unfamiliar technical language as gibberish can discard useful observations;
missing a sensitive identifier can leave material requiring further review.
A configurable first pass can make these decisions explicit and reproducible
without requiring a remote inference service.

The accompanying research workflow evaluates published human-produced
gibberish and English comparison texts [@gaskell2022]. It records source
classifications, detector scores, applicability and false flags, and compares
fixed profiles with a pinned local DistilBERT classifier and retains an
earlier character-model calibration experiment. This provides a concrete use of
pygarble for investigating the limits of inexpensive text filters. The shared
analysis interface makes configuration differences inspectable without
requiring remote inference. It does not supply contextual semantic judgments.

The contribution is a maintained, configurable library and reproducible
workflow for composing local screening policies. The independent performance
of each component remains a separate question; this evaluation does not
establish the research performance of PII, secret or profanity screening.

# State of the field

Existing projects address related parts of the problem. Presidio offers
extensible PII detection and de-identification, including pattern-based and
NLP recognizers [@presidio]. Gitleaks and detect-secrets focus on finding
credentials [@gitleaks; @detectsecrets]. The phonenumbers library parses and
validates telephone numbers, while python-stdnum supplies parsers and
validators for standardized identifiers [@phonenumbers; @stdnum].
better_profanity supports word-list censorship and modified spellings
[@betterprofanity]. Gibberish-Detector illustrates character-transition
modeling for gibberish classification [@gibberishdetector].

pygarble reuses phonenumbers, python-stdnum, detect-secrets and Gitleaks
through explicit optional adapters. Its native rules provide a dependency-free
baseline; the shared finding and redaction representation lets applications
combine selected tools. This is a design rationale for integration, not a
claim that individual rules are novel or that existing systems cannot support
similar workflows. Presidio's broader entity recognition and specialized
secret scanners remain relevant alternatives. A narrow rule set also trades
contextual coverage for simplicity: word matching does not assess toxicity,
and structured-identifier checks do not recognize every personal reference.

The accompanying comparison includes single-strategy references and an
external pretrained DistilBERT classifier [@jindal]. The preliminary study
also used locally trained character bigram/trigram baselines. Word lookup
detected more positive documents than the default ensembles in this corpus. This finding argues
for explicit configuration and evaluation, not a claim of ensemble superiority.
The software's significance beyond this workflow remains for author and
editorial assessment; combining existing tools alone does not establish it.

# Software design

The architecture separates the screening and gibberish modules while
preserving earlier import paths. This separation lets sensitive-content
scanning avoid loading the gibberish ensemble and its resources. Shared
finding types retain original character offsets and rule reasons, and
redaction combines overlapping qualifying spans before replacement. Native
PII checks combine candidate patterns with applicable checksums; profanity
checks normalize selected obfuscations; secret checks combine known formats
and contextual heuristics. Their outputs are evidence tiers, not calibrated
probabilities or confirmation that a credential is active.

Gibberish detection combines configurable strategies through profiles and
voting policies. An applicability signal distinguishes unavailable evidence
from a clean result. Threshold calibration supports choosing a decision
boundary against labelled examples. English-oriented dictionaries and rules
restrict the interpretation of those scores; valid identifiers, specialist
vocabulary and other languages require appropriate configuration and testing.

Optional backends are selected explicitly. Installing an extra does not
silently change detection behavior. Python integrations run in process;
Gitleaks runs as a local subprocess per document, which introduces launch
cost and a different resource-management boundary. Its byte offsets must be
translated to Python character offsets before redaction. Backend failures
raise errors rather than produce an apparently clean scan. These choices
favor predictable composition while leaving deployment-specific limits and
failure handling to the calling application.

# Research impact statement

Trident [@saul2026trident, Section 5.3] cites pygarble as a possible
postprocessing tool for distinguishing random-looking filenames from
non-random matches in malware-detection rules. The authors explicitly state
that their evaluated approach applies no postprocessing. This citation
therefore documents independent recognition of a potential application,
not deployment or a measured contribution to Trident's results. It concerns
gibberish detection and provides no evaluation of the newer screening modules.

An author-requested, AI-assisted external evaluation has now been executed
using the published corpus [@gaskell2022]. Its frozen protocol, source hashes,
code, predictions and report are available in
[paper/study](https://github.com/brightertiger/pygarble/tree/main/paper/study).
The full experiment processes every released labelled document: 38 gibberish
transcripts and 71 meaningful texts, comprising 79,969 consecutive chunks.
English controls are primary; the other 67 meaningful documents are reported
separately. Under a fixed majority-chunk rule, English detects 13 positive
documents, while word lookup and both transformer label policies detect all
38. Word lookup flags none of the 5,200 English control chunks; the two
transformer policies differ sharply in false flags. Separate preliminary
source-held-out calibration exposed substantial false-positive shifts.
These are collection-specific measurements, not population accuracy claims.

The workflow retains supplied classifications without a new human label
audit. Limited source diversity, historical spelling, source/class confounding
and unknown participant dependencies constrain interpretation. Automated
checks verify full-corpus artifacts and replay a fixed neural subset offline.
A separate fresh-environment rerun reproduced the preliminary deterministic
results byte for byte. Neither constitutes independent human reproduction
or manuscript review.
The study establishes an executed research workflow for author review,
without claiming external adoption, downstream improvement or automatic JOSS
eligibility. The broader repository supplies tests, CI, example workflows
and regression corpora to support continued maintenance and reuse.

# AI usage disclosure

The author reports using OpenAI Codex and Anthropic Claude. Codex assisted
with repository development and documentation work,
including module organization, optional backend integration, validation work
and preparation of this manuscript. Codex also assisted with the published-
corpus study design, implementation, execution, analysis and draft text. This drafting session uses GPT-6.
Automated tests and CI provide implementation checks; they do not replace
human review of the manuscript or scientific claims.

Claude assisted with coding and running comparison experiments; exact model
versions were not recorded. The author must verify this disclosure. After personally reviewing
and validating the material, confirm human responsibility for core design
decisions and all AI-assisted outputs. That confirmation is not asserted here
on the authors' behalf.

# Acknowledgements

The author confirms that this work received no funding and declares no
conflicts of interest. The author is the developer of pygarble.

# References
