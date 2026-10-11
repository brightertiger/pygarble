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
- **2026-10-11: checkpoint fired; run continued without amendment.** After
  the B dev cuts were fitted, `english_fusion` captured 0.600 of A at the
  `fpr1` cut, below the 0.70 floor, and the runner stopped before B test.
  The maintainer's standing instruction was to run the benchmark end to
  end. The orchestrator chose to continue under the frozen protocol with no
  amendment: B test was scored once with `run.py --override-checkpoint`,
  and the override is recorded in `results/checkpoint.json`. Fusion's
  0.600 is reported as a loss.
- **Ties at the top score (reporting only).** For some methods more than
  `k` dev tweets share the top score, so the frozen cut equals the highest
  score of any B dev row and flags no B dev row. The cut rule is
  unchanged; these points carry `reachable: false` in `metrics.json` and
  a dagger on the cut in `results.md`. Their A and B test numbers are
  still reported, because a few A or B test rows can score above the
  highest B dev score (for example `english_vote_all` flags 4 B test rows
  at both cuts). Fitted cuts and dev FPR are stored with each operating
  point.

- **2026-10-11: benchmark narrowed to default decisions.** At the
  maintainer's direction, the matched-FPR operating points (cuts at 1 and
  0.1 percent FPR on B dev tweets), the 30/70 development/test split of B,
  the checkpoint and its override, the calibration run, AUROC and the
  precision-recall curves computed in PR 18 were removed from the
  benchmark. They remain only in git history (merge commit `b968953`). B is
  now scored whole at each method's default decision. The entries above
  on the checkpoint, the cut ties and the calibration positives describe
  that removed machinery and are kept as a record.
- **2026-10-11: methods reduced to three.** At the maintainer's direction
  the comparison is pygarble, gibberish-detector and lingua only. nostril,
  gibberish-chat-detector, the Gopher and C4 rules, the five other
  pygarble profiles, the six voting modes and the single-strategy appendix
  were removed from the runner, results and paper, with their wrappers,
  `rules.py` and their requirements. Their PR 18 results remain in git
  history. pygarble is reported as two rows: the default `english`
  profile and `english_fusion`. The entries above on nostril, the shared
  module name and the Gopher rules describe removed baselines.
- **2026-10-11: post hoc tuned row added.** At the maintainer's direction
  a fifth row, `english_tuned`, is the `english` profile without its
  mojibake and control-character members (same `any` voting, threshold
  0.5). It was constructed after the results showed that 26.4 percent of B
  tweets carry mojibake, which B labels clean, and that these two
  encoding-damage members flag them. It is a domain-specific tuning, not a
  pre-registered result, and is reported separately as such. The
  `english_fusion` profile has no mojibake member, so it has no variant.

- **2026-10-11: weight tuning study added.** At the maintainer's request
  a tuning study (`paper/scripts/benchmark/tune.py`, results in
  `results/tuning/`) learns a weight in [0, 1] for each of pygarble's 32
  strategies and a threshold in [0.05, 0.8] under `weighted` voting, with
  Optuna's TPE sampler (1,000 trials per draw, F1 objective, junk
  positive). For each
  of five seeds (20261011 to 20261015) the tuning sample is a stratified
  20 percent of A and 20 percent of B by label; evaluation is on the
  held-out 80 percent, where the `english` and `english_fusion` profiles
  are scored on the same rows. The study uses labelled data from both
  datasets, so the tuned rows are post hoc in that sense, but no
  evaluation row is seen during tuning. It replaces the hand-made
  `english_tuned` variant (the `english` profile without its mojibake and
  control-character members) described in the previous entry. That
  variant's predictions, metrics, footprint entry and preprint rows were
  removed; they remain in git history.
- **2026-10-11: tuning search widened.** The first tuning run used a
  threshold range of [0.2, 0.8] and 400 trials per draw. In every one of
  the five draws the chosen threshold sat on the 0.2 bound, and the best
  trial came late (trials 237 to 395 of 400), so the search had not
  converged. On that evidence from the tuning samples, the maintainer
  widened the threshold range to [0.05, 0.8] and raised the budget to
  1,000 trials per draw. `tune.py` always evaluates on the held-out rows,
  so the first run's held-out results had also been computed (tuned
  weights, mean over five draws: tweet FPR 0.028, survey-junk capture
  0.833, random-string recall 0.922); the widened run has a higher tweet
  FPR (0.047) and higher capture and recall. Even with 1,000 trials the
  best trial came late (930 to 996), so the search was still improving.
  Seeds, the 20 percent sample, the objective and the search grid are
  unchanged. The widened run is the one reported; the first run's results
  remain in git history (commit `9f213ef`).

No other method was added or changed.
