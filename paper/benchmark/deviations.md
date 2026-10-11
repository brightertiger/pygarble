# Deviations from the plan

Departures from `paper/PREPRINT-PLAN.md` (v4) and decisions the plan left
open. Each is also stated in `protocol.md` where it affects the protocol.

- **nostril source.** PyPI `nostril` 0.1.1 is Pedro Paiva's nose wrapper,
  not Hucka's Nonsense String Evaluator. Hucka's nostril is installed from
  GitHub `casics/nostril` at commit
  `fbc0c91249283a9fbc9036206391ce1138826fd3`.
- **gibberish-detector model.** The 0.1.1 wheel ships no model or corpus.
  The model is trained on `examples/big.txt` from
  `domanchi/gibberish-detector` at commit
  `ecac96918e72fbf702cfee3274dbf280009171ad` (sha256
  `e77a1d4aeefb1f2c32fe0fc30551478a7eb59dc093f270c095822290971d39d7`),
  the corpus its README trains on.
- **Shared module name.** gibberish-detector and gibberish-chat-detector
  both install a top-level `gibberish_detector` package. Their submodules
  (`detector`, `trainer` and `gibberish`) do not overlap, so both are
  installed in one venv and imported by submodule.
- **Gopher rules.** The duplicate-line and duplicate-paragraph rules have no
  RedPajama-V2 signal and are omitted. The stop-word rule follows Gopher
  (at least two of eight words), not RedPajama-V2's stop-word fraction.
- **Weighted voting.** pygarble ships no weights for the `english` profile,
  so `weighted` uses equal weights and matches `average` in score.
- **B duplicates kept.** B is not de-duplicated, matching the 4,000 and
  5,971 counts in the plan; 132 B test rows share their text with a B dev
  row.
- **Calibration positives.** `calibrate()` needs junk examples as well as
  clean ones; 500 B dev random strings are used alongside the 500 B dev
  tweets.

No baseline was dropped.
