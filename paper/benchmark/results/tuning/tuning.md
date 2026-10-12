# Weight tuning study

Tuning sample: a stratified 20% of A and of B per seed, drawn by unit so that rows with identical text stay on the same side of the split; Optuna 5.0.0 TPE, 1000 trials, objective F1.
Held-out rows (primary seed 20261011): A 2810, B 7977 (3200 random strings, 4777 tweets).

## Held-out metrics, primary seed 20261011 (95% bootstrap CI)

| method | A capture | B precision | B recall | B FPR |
|---|---|---|---|---|
| pygarble (english) | 0.826 [0.811, 0.840] | 0.653 [0.639, 0.666] | 0.949 [0.941, 0.956] | 0.338 [0.324, 0.351] |
| pygarble (english_fusion) | 0.891 [0.879, 0.903] | 0.904 [0.894, 0.913] | 0.990 [0.987, 0.994] | 0.071 [0.064, 0.078] |
| pygarble (tuned weights) | 0.868 [0.856, 0.882] | 0.941 [0.934, 0.949] | 0.967 [0.960, 0.973] | 0.040 [0.035, 0.046] |

## Held-out metrics, mean ± sd over 5 seeds

| method | A capture | B precision | B recall | B FPR |
|---|---|---|---|---|
| pygarble (english) | 0.830 ± 0.005 | 0.653 ± 0.002 | 0.946 ± 0.002 | 0.337 ± 0.003 |
| pygarble (english_fusion) | 0.896 ± 0.004 | 0.906 ± 0.002 | 0.991 ± 0.001 | 0.069 ± 0.002 |
| pygarble (tuned weights) | 0.897 ± 0.027 | 0.930 ± 0.016 | 0.976 ± 0.023 | 0.049 ± 0.012 |

## Learned parameters

| seed | threshold | members | tuning F1 | best trial | seconds |
|---|---|---|---|---|---|
| 20261011 | 0.16 | 29 | 0.951 | 952 | 17.0 |
| 20261012 | 0.14 | 29 | 0.950 | 993 | 17.4 |
| 20261013 | 0.15 | 24 | 0.958 | 948 | 17.0 |
| 20261014 | 0.15 | 31 | 0.931 | 992 | 17.5 |
| 20261015 | 0.13 | 27 | 0.965 | 832 | 16.9 |

Total tuning time 85.9 s; scoring the cache took 12.7 s.

| strategy | mean weight | primary-seed weight |
|---|---|---|
| word_lookup | 1.00 | 1.00 |
| zipf_conformity | 0.81 | 0.25 |
| entropy_based | 0.78 | 1.00 |
| hex_string | 0.76 | 0.85 |
| permutation_test | 0.73 | 0.25 |
| letter_position | 0.71 | 0.25 |
| log_likelihood_ratio | 0.69 | 0.90 |
| word_anomaly | 0.67 | 1.00 |
| ngram_frequency | 0.60 | 0.65 |
| vowel_pattern | 0.57 | 0.65 |
| primed_compression | 0.52 | 0.95 |
| bigram_probability | 0.51 | 0.55 |
| consonant_sequence | 0.51 | 0.50 |
| unicode_script | 0.50 | 0.45 |
| keyboard_adjacency | 0.48 | 0.60 |
| markov_chain | 0.48 | 0.80 |
| repetition | 0.42 | 0.30 |
| affix_detection | 0.41 | 0.15 |
| rare_trigram | 0.40 | 0.70 |
| function_word_density | 0.39 | 0.45 |
| symbol_ratio | 0.38 | 0.85 |
| letter_frequency | 0.34 | 0.75 |
| local_anomaly | 0.33 | 0.05 |
| cross_parsing | 0.31 | 0.60 |
| pattern_matching | 0.30 | 0.00 |
| keyboard_pattern | 0.28 | 0.35 |
| pronounceability | 0.28 | 0.65 |
| word_collocation | 0.25 | 0.05 |
| vowel_ratio | 0.23 | 0.05 |
| ngram_rank | 0.10 | 0.05 |
| control_characters | 0.04 | 0.00 |
| mojibake | 0.03 | 0.00 |
