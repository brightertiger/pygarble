# Full released corpus and Hugging Face comparison

Extension fixed on 27 September 2026 before its detector outcomes. The author
requested the complete published GitHub dataset, a Hugging Face benchmark
and pygarble. Earlier capped results are already known; this extension is
exploratory and is not a new untouched confirmation set. Original results
and protocol remain unchanged for auditability.

## Dataset and units

Use every released transcript in `gibberish_transcriptions.zip` (38 files)
and every `texts/*.txt` member of `meaningful.zip` (71 files), at the same
pinned Gaskell/Bowern revision recorded in `sources.json`. Exclude only archive
metadata and `meaningful_sources.txt`. Do not use `voynichese.zip` as labelled
binary data: its classification is unresolved. Thus "full corpus" means the
full released labelled meaningful/gibberish collection, not the original
paper's unreleased materials or an invented Voynich label.

Normalize whitespace exactly as before. Partition each complete document
into consecutive, nonoverlapping 400-character chunks, retaining its final
short chunk. No sampling cap or text is dropped after normalization. This
produces 79,969 chunks from 109 documents (173 gibberish and 79,796 meaningful).
The original 400-character prefixes are contained in this extension; repeated
text does not become new independent evidence. Never count 79,969 chunks as
79,969 independent documents. Preserve document, language, era and genre IDs.

English is the primary target: all chunks from the four explicitly English
meaningful documents versus all 38 gibberish documents. The other 67 meaningful
documents cover other natural/historical/constructed languages and form a
separate out-of-scope false-flag analysis. Their meaningful labels remain
negative; do not call a valid foreign-language document gibberish merely
because an English detector rejects it. Report every language/source without
pooling this diagnostic with the primary English headline.

No new human annotation or label audit. The sources' labels are inherited,
including on short final fragments. Separate full-width and tail-chunk counts
and report sensitivity excluding tails. No raw comparison prose is republished.

## Models and policies

Use pygarble's fixed English, extended and legacy profiles and word-lookup
and entropy strategies, all at their original 0.5 defaults. No runtime-library
changes or tuning. The character baselines and calibration experiment from
the preliminary study remain documented separately.

Add `madhurjindal/autonlp-Gibberish-Detector-492513457` at immutable revision
`76672dd7d3575f68ab980705bcec975cc62de71c`: a 66,956,548-parameter DistilBERT
classifier. Pin model/tokenizer hashes in `hf_model.json`. Load only local
verified safetensors and ordinary tokenizer/config files; no remote code.
Its card declares MIT, but the named training dataset's public API returns
401. Fine-tuning overlap with this corpus cannot be ruled out. The card's
advertised validation accuracy is not our benchmark result.

Use all four output labels as supplied. Primary HF decision: top-probability
class is anything other than `clean`. Secondary conservative policy: the
winning class is `noise` or `word salad`; `mild gibberish` is not flagged.
Report both policies regardless of results. Keep every class probability,
winning label and token count. Never replace the source labels with model
predictions. The binary score `1 - P(clean)` is descriptive, not a calibrated
probability of our task. No fine-tuning or test-based threshold selection.

Use the complete 400-character chunk for both methods. Tokenize with the
model's normal special tokens and dynamic padding; assert no input exceeds
512 tokens and do not silently truncate. Full precision, evaluation mode,
no gradients, CPU, four intra-op and one inter-op thread, batch size 8,
PyTorch scaled-dot-product attention (`sdpa`). Tokenizer
parallelism is disabled. If hardware/library behavior requires a change,
record it before the affected run and retain the prior protocol.

## Endpoints

Primary document decision: at least half of its chunks are flagged (ties
flagged). This is an explicit study aggregation rule, not the model's native
long-document interface. Report positive-document recall, false flags out of
four English control documents, and each document's flagged-chunk fraction.
Also report macro mean flagged fraction over positive documents, macro FPR
over English documents and descriptive micro chunk counts. Do not present
prevalence-dependent accuracy/F1 as general quality measures.

For every document, report both any-chunk and majority-chunk decisions as
sensitivity diagnostics; any-chunk flags naturally grow with document length.
Report tail-excluded macro rates separately. Uncertainty uses the original
Wilson function on the 38 positive document decisions, conditional on document
independence. Use the original 2,000-replicate paired document bootstrap for
recall differences versus the HF primary policy. No passage-level confidence
interval or multiple-testing significance claim. Language-level negative
rates retain source/document counts and are descriptive only.

Store compressed predictions per source document, and an unhashed-text-free
record manifest with normalized offsets, lengths and text hashes. Check that
chunks reconstruct each normalized document exactly and that every expected
archive member is included once. Report exact duplicate chunk occurrences
without silently dropping them. Record hash overlap with the preliminary
400-character sample to make reuse visible. Related Bible translations and
abbreviated variants remain dependent sources; no new random split is made.

## Runtime, validation and paper

Download model weights once (about 268 MB); inference is offline with no
external endpoint or fee. Run an end-to-end CPU timing comparison on the
original fixed 76 primary inputs, batch size 1, including tokenization for
HF and the full score path for pygarble. Re-measure both in this session,
with one warm pass and five timed passes, and one intra-op/inter-op thread
for HF timing to match the single-thread baseline comparison. Run fresh
processes per method;
measure imports/load, first call and peak process RSS separately. Disk cache
and model download time are excluded from inference timing. Report model
weight bytes and dependencies so the cost difference is visible.

Verify hashes, probability sums, label mapping, complete corpus coverage,
recomputed summaries and saved-prediction membership. Replay first/middle/last
chunks of every document in a new offline process with the same batch size;
allow probability differences up to 1e-5 from batch padding, but require
identical policy decisions. Unit tests must cover majority/ties, source
membership, tails, label mapping and normalized text reconstruction.

Update the empirical manuscript around this full-corpus comparison, preserve
the preliminary results as such, and regenerate the review PDF/source bundle.
Keep all additions under `paper/study/` and push through the current PR.
No paper submission, merge, model integration into the base package or new
human label audit is authorized by this experiment. Author review follows.
