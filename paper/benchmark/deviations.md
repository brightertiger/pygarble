# Deviations from the plan

Departures from the benchmark plan as first written (2026-10-11, since
superseded by `protocol.md`) and decisions the plan left open. Each is also stated in `protocol.md` where it affects the protocol.

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

- **2026-10-12: grouped split.** B keeps 215 rows that repeat an
  earlier text (86 groups, all tweets, the largest with 48 copies), and
  the tuning sample was drawn row by row, so copies of one tweet could sit
  in both the tuning sample and the held-out set. `draw()` in `tune.py`
  now samples by unit, a unit being a duplicate group from the manifest
  or a single row, stratified by label, with the same seeds; no group is
  split and the sample sizes (703 A rows, 800 random strings, 1,194
  tweets) are unchanged. No rows were deleted and the main
  default-decision results are untouched. The full tuning was re-run (five
  draws, 1,000 trials, same search space). Held-out results before and
  after:

  | method | numbers | capture A | precision B | recall B | FPR B |
  |---|---|---|---|---|---|
  | tuned | primary, row split | 0.899 | 0.923 | 0.988 | 0.055 |
  | tuned | primary, grouped | 0.868 | 0.941 | 0.967 | 0.040 |
  | tuned | mean, row split | 0.895 | 0.933 | 0.981 | 0.047 |
  | tuned | mean, grouped | 0.897 | 0.930 | 0.976 | 0.049 |
  | english_fusion | primary, row split | 0.893 | 0.907 | 0.992 | 0.068 |
  | english_fusion | primary, grouped | 0.891 | 0.904 | 0.990 | 0.071 |
  | english_fusion | mean, row split | 0.894 | 0.907 | 0.991 | 0.068 |
  | english_fusion | mean, grouped | 0.896 | 0.906 | 0.991 | 0.069 |
  | english | primary, row split | 0.828 | 0.657 | 0.949 | 0.332 |
  | english | primary, grouped | 0.826 | 0.653 | 0.949 | 0.338 |
  | english | mean, row split | 0.830 | 0.656 | 0.949 | 0.333 |
  | english | mean, grouped | 0.830 | 0.653 | 0.946 | 0.337 |

  The tuned configuration still has a lower tweet FPR and higher precision
  than `english_fusion` in every draw, but one conclusion changed: recall
  is now 1.5 points lower on average (it was 1.0) and lower in four of five
  draws, and on the primary draw the tuned capture and recall are now
  below the profile's, so the paper no longer says tuning keeps capture
  and recall within about a point and describes it as a trade of recall
  for precision. The row-split results remain in git history (main at
  `7047096`).

No other method was added or changed.
