# Weight tuning study

Tuning sample: a stratified 20% of A and of B per seed; Optuna 5.0.0 TPE, 1000 trials, objective F1.
Held-out rows (primary seed 20261011): A 2810, B 7977 (3200 random strings, 4777 tweets).

## Held-out metrics, primary seed 20261011 (95% bootstrap CI)

| method | A capture | B precision | B recall | B FPR |
|---|---|---|---|---|
| pygarble (english) | 0.828 [0.816, 0.841] | 0.657 [0.642, 0.671] | 0.949 [0.942, 0.956] | 0.332 [0.319, 0.345] |
| pygarble (english_fusion) | 0.893 [0.881, 0.904] | 0.907 [0.897, 0.916] | 0.992 [0.988, 0.994] | 0.068 [0.061, 0.075] |
| pygarble (tuned weights) | 0.899 [0.889, 0.911] | 0.923 [0.914, 0.933] | 0.988 [0.984, 0.991] | 0.055 [0.049, 0.062] |

## Held-out metrics, mean ± sd over 5 seeds

| method | A capture | B precision | B recall | B FPR |
|---|---|---|---|---|
| pygarble (english) | 0.830 ± 0.002 | 0.656 ± 0.001 | 0.949 ± 0.002 | 0.333 ± 0.001 |
| pygarble (english_fusion) | 0.894 ± 0.001 | 0.907 ± 0.002 | 0.991 ± 0.001 | 0.068 ± 0.001 |
| pygarble (tuned weights) | 0.895 ± 0.022 | 0.933 ± 0.017 | 0.981 ± 0.017 | 0.047 ± 0.012 |

## Learned parameters

| seed | threshold | members | tuning F1 | best trial | seconds |
|---|---|---|---|---|---|
| 20261011 | 0.13 | 28 | 0.952 | 948 | 17.6 |
| 20261012 | 0.17 | 29 | 0.958 | 968 | 17.6 |
| 20261013 | 0.18 | 26 | 0.950 | 971 | 17.9 |
| 20261014 | 0.15 | 28 | 0.938 | 930 | 17.6 |
| 20261015 | 0.18 | 25 | 0.973 | 996 | 17.7 |

Total tuning time 88.3 s; scoring the cache took 12.7 s.

| strategy | mean weight | primary-seed weight |
|---|---|---|
| word_lookup | 1.00 | 1.00 |
| ngram_frequency | 0.74 | 0.90 |
| log_likelihood_ratio | 0.71 | 0.95 |
| hex_string | 0.67 | 0.90 |
| permutation_test | 0.67 | 0.15 |
| vowel_pattern | 0.62 | 0.85 |
| zipf_conformity | 0.61 | 0.80 |
| entropy_based | 0.58 | 0.70 |
| function_word_density | 0.58 | 0.90 |
| word_anomaly | 0.56 | 0.90 |
| consonant_sequence | 0.54 | 0.70 |
| letter_position | 0.50 | 0.15 |
| rare_trigram | 0.46 | 0.95 |
| markov_chain | 0.45 | 0.35 |
| letter_frequency | 0.42 | 0.75 |
| symbol_ratio | 0.40 | 0.65 |
| unicode_script | 0.40 | 0.85 |
| primed_compression | 0.39 | 0.40 |
| cross_parsing | 0.35 | 1.00 |
| repetition | 0.34 | 0.10 |
| local_anomaly | 0.29 | 0.35 |
| affix_detection | 0.28 | 0.05 |
| ngram_rank | 0.28 | 0.00 |
| keyboard_adjacency | 0.26 | 0.35 |
| keyboard_pattern | 0.26 | 0.05 |
| word_collocation | 0.23 | 0.10 |
| pattern_matching | 0.21 | 0.05 |
| vowel_ratio | 0.17 | 0.10 |
| bigram_probability | 0.15 | 0.25 |
| pronounceability | 0.15 | 0.00 |
| mojibake | 0.06 | 0.00 |
| control_characters | 0.00 | 0.00 |
