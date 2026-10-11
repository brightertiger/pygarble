# Weight tuning study

Tuning sample: a stratified 20% of A and of B per seed; Optuna 5.0.0 TPE, 400 trials, objective F1.
Held-out rows (primary seed 20261011): A 2810, B 7977 (3200 random strings, 4777 tweets).

## Held-out metrics, primary seed 20261011 (95% bootstrap CI)

| method | A capture | B precision | B recall | B FPR |
|---|---|---|---|---|
| pygarble (english) | 0.828 [0.816, 0.841] | 0.657 [0.642, 0.671] | 0.949 [0.942, 0.956] | 0.332 [0.319, 0.345] |
| pygarble (english_fusion) | 0.893 [0.881, 0.904] | 0.907 [0.897, 0.916] | 0.992 [0.988, 0.994] | 0.068 [0.061, 0.075] |
| pygarble (tuned weights) | 0.851 [0.839, 0.864] | 0.961 [0.954, 0.968] | 0.946 [0.938, 0.953] | 0.026 [0.021, 0.030] |

## Held-out metrics, mean ± sd over 5 seeds

| method | A capture | B precision | B recall | B FPR |
|---|---|---|---|---|
| pygarble (english) | 0.830 ± 0.002 | 0.656 ± 0.001 | 0.949 ± 0.002 | 0.333 ± 0.001 |
| pygarble (english_fusion) | 0.894 ± 0.001 | 0.907 ± 0.002 | 0.991 ± 0.001 | 0.068 ± 0.001 |
| pygarble (tuned weights) | 0.833 ± 0.022 | 0.957 ± 0.012 | 0.922 ± 0.035 | 0.028 ± 0.009 |

## Learned parameters

| seed | threshold | members | tuning F1 | best trial | seconds |
|---|---|---|---|---|---|
| 20261011 | 0.20 | 28 | 0.939 | 237 | 5.4 |
| 20261012 | 0.20 | 30 | 0.920 | 395 | 5.4 |
| 20261013 | 0.20 | 28 | 0.921 | 386 | 5.7 |
| 20261014 | 0.20 | 28 | 0.935 | 277 | 5.7 |
| 20261015 | 0.20 | 31 | 0.903 | 370 | 5.8 |

Total tuning time 28.0 s; scoring the cache took 12.3 s.

| strategy | mean weight | primary-seed weight |
|---|---|---|
| word_lookup | 0.98 | 1.00 |
| ngram_frequency | 0.83 | 0.95 |
| letter_position | 0.77 | 0.95 |
| ngram_rank | 0.70 | 0.95 |
| log_likelihood_ratio | 0.68 | 0.80 |
| entropy_based | 0.67 | 0.85 |
| hex_string | 0.67 | 0.80 |
| markov_chain | 0.66 | 0.90 |
| vowel_ratio | 0.58 | 0.80 |
| word_anomaly | 0.57 | 0.25 |
| primed_compression | 0.56 | 0.75 |
| rare_trigram | 0.54 | 0.70 |
| symbol_ratio | 0.54 | 0.55 |
| pronounceability | 0.53 | 0.50 |
| unicode_script | 0.53 | 0.45 |
| zipf_conformity | 0.51 | 0.65 |
| word_collocation | 0.49 | 0.20 |
| local_anomaly | 0.47 | 0.75 |
| affix_detection | 0.43 | 0.50 |
| cross_parsing | 0.41 | 0.00 |
| permutation_test | 0.41 | 0.15 |
| vowel_pattern | 0.39 | 0.05 |
| pattern_matching | 0.37 | 0.00 |
| keyboard_pattern | 0.35 | 0.65 |
| letter_frequency | 0.35 | 0.10 |
| function_word_density | 0.34 | 0.20 |
| consonant_sequence | 0.32 | 0.20 |
| bigram_probability | 0.29 | 0.00 |
| mojibake | 0.28 | 0.40 |
| repetition | 0.28 | 0.60 |
| keyboard_adjacency | 0.19 | 0.10 |
| control_characters | 0.07 | 0.00 |
