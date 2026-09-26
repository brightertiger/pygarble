Choosing strategies
===================

``EnsembleDetector()`` selects the default English screening profile. It combines three English
language heuristics with mojibake, keyboard adjacency, and control-character
checks. Meaningful Hindi may be flagged; this is expected behavior.

The ``english_extended`` profile adds localized anomalies, repetition, and pattern
matching. Measure its false positives before using it for automatic rejection.
Use ``corruption`` for encoding/control artifacts without English plausibility
scoring. ``spoofing`` is a Unicode script/confusable heuristic, not a full phishing
or Unicode security implementation. Profile membership is listed in :doc:`strategies`.

What the paper tells us about strategy choice
---------------------------------------------

The published-corpus evaluation in :doc:`benchmarks` compares the three English
profiles, word lookup and entropy with two fixed DistilBERT label policies.
Word lookup made no errors on the 173 gibberish and 5,200 English-control chunks,
while the default English profile caught only 49 gibberish chunks. This is
specific to the invented-text collection; it does not justify replacing all
other strategies or claiming universal 100% accuracy. The profiles also target
encoding, keyboard and structural defects that this collection does not
comprehensively represent.

Lexical and statistical strategies
----------------------------------

* ``word_lookup`` folds diacritics and checks Latin-letter words of at least
  two letters against the embedded dictionary. Unknown title-cased words
  contribute half weight. The default score is the weighted unknown fraction;
  a score of at least 0.5 flags the input. No eligible tokens yields zero,
  which is not evidence of linguistic understanding. Word lookup is not a
  member of the named English profiles.
* ``markov_chain`` measures English character-transition likelihood over
  novel tokens. Dictionary-supported and recognized structured tokens do
  not contribute novelty evidence. Unlikely transitions increase the score.
* ``log_likelihood_ratio`` compares English bigram likelihood with a uniform
  reference. Its sigmoid maps that statistic to a heuristic score, not a
  calibrated probability.
* ``word_anomaly`` scores suspicious eligible tokens individually and reports
  their fraction, limiting dilution by surrounding ordinary prose.
* ``entropy_based`` combines character entropy and bigram improbability.
  It detected no positive chunks at its default threshold in this study.

Scores are strategy-specific evidence. Named English ensembles use ``any``
voting across applicable strategies; an inapplicable strategy does not vote.
When all strategies are inapplicable, ``analyze()`` reports
``insufficient_evidence`` and an unflagged decision. That differs from a
positive assertion that the text is meaningful. See :doc:`api` for other
voting policies and score semantics, and :doc:`calibration` for threshold
selection on your own labelled data.

Control characters: new in 0.9.0
--------------------------------

``Strategy.CONTROL_CHARACTERS`` examines raw Unicode characters before English
normalization. It detects unexpected control characters, U+FFFD replacement
characters, lone surrogates, and combining-mark runs exceeding a configurable
limit. Ordinary tabs, line breaks, accents, and emoji joiners are not themselves
flagged. This strategy belongs to ``english``, ``english_extended``,
``corruption``, and ``llm_output``.

.. code-block:: python

   from pygarble.gibberish import GarbleDetector, Strategy

   detector = GarbleDetector(
       Strategy.CONTROL_CHARACTERS,
       strategy_kwargs={"max_combining_run": 8},
   )
   assert detector.predict("hello\x00world") is True
   assert detector.predict("hello\nworld") is False
   assert detector.predict("hello\ufffdworld") is True

``max_combining_run`` defaults to 8 and must be a positive integer. A detected
artifact scores 0.9; otherwise the score is 0.0. A detector threshold above 0.9
therefore suppresses these detections. These are fixed heuristic scores.

Localized anomalies: new in 0.9.0
---------------------------------

``Strategy.LOCAL_ANOMALY`` finds severe unfamiliar tokens or clusters of suspicious
tokens in otherwise readable English. It uses the bundled English dictionary,
shared bigram likelihood, and bounded token windows. It belongs to
``english_extended`` and ``llm_output`` and can also be selected independently.

.. code-block:: python

   from pygarble.gibberish import GarbleDetector, Strategy

   detector = GarbleDetector(
       Strategy.LOCAL_ANOMALY,
       strategy_kwargs={
           "min_word_length": 8,
           "word_log_prob_threshold": -5.5,
           "window_words": 4,
       },
   )
   text = "Please review qxzjkwpvm before delivery."
   result = detector.analyze(text)
   assert result.garbled is True
   assert any(text[s.start:s.end] == "qxzjkwpvm" for s in result.spans)

The example shows the defaults. ``min_word_length`` is the minimum length for an
individual severe-token span; windows can use suspicious tokens of at least four
characters. ``word_log_prob_threshold`` must be finite and negative; a more
negative value requires a lower likelihood. ``window_words`` must be an integer
from 1 to 32. A window requires at least two suspicious tokens, so choosing 1
disables window detections while retaining individual-token checks.

Detected anomalies score 0.8. Dictionary words, allowlisted terms, and structured
tokens excluded by shared preprocessing do not become suspicious candidates.
This is not a grammaticality or semantic-coherence check, and rare English words
can still produce false positives.

Keyboard paths and repetition
-----------------------------

Keyboard adjacency supports QWERTY, AZERTY, and QWERTZ with digit-row neighbors.
It checks straight rows and substantially covered adjacent-key paths. It does not
detect every possible keyboard walk. Repetition detects character, word, and
bounded phrase repetition; intentional repetition can also be flagged.

.. code-block:: python

   from pygarble.gibberish import GarbleDetector, Strategy

   keyboard = GarbleDetector(
       Strategy.KEYBOARD_ADJACENCY, keyboard_layout="azerty"
   )
   assert keyboard.predict("azerty") is True

   repetition = GarbleDetector(Strategy.REPETITION)
   assert repetition.predict("hello " * 10) is True

LLM output: new in 0.10.0
-------------------------

The ``llm_output`` profile is a cheap deterministic pre-check for degenerate
model output, not a hallucination detector. It combines repetition, control
characters, mojibake and local anomaly, so it catches repetition loops,
encoding damage from bad decoding, stray control characters and dense token
salad. It deliberately leaves out the Markov and word-anomaly members, so code,
identifiers, product names and technical prose stay quiet. Use it as a guard
that retries or rejects a response before it reaches a user.

.. code-block:: python

   from pygarble.gibberish import EnsembleDetector

   guard = EnsembleDetector(profile="llm_output")
   assert guard.predict("and so on and so on and so on and so on") is True
   assert guard.predict("The answer is CafÃ© au lait.") is True
   assert guard.predict('def parse(row): return row.split(",")') is False

   response = "Restart the service, then confirm the new log file appears."
   if guard.predict(response):
       raise RuntimeError("degenerate response; retry the request")

Tune on your own data
---------------------

All 28 strategies remain individually available. Check both valid English inputs
and expected corruption, including domain terms, identifiers, and short strings.
Scores are not calibrated probabilities, and no strategy guarantees zero false
positives. See :doc:`api` for voting and abstention, and :doc:`migration` for changes
from the previous default. The earlier conditional-trigram experiment is retained in Git history;
no additional trigram model is shipped by 0.10.0.
