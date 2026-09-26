# Source notices

Research code follows the repository MIT license. Dataset rights are separate.
Raw data are fetched from the pinned upstream archive for local analysis;
comparison texts and extracted passages are not redistributed in this folder.
The archive's `meaningful_sources.txt` records original sources and conditions.
The four English controls include KJV, NET Bible, Nico Koomen's transcription
of Secreta Alberti, and Wikipedia's Voynich article. Do not apply the repository
MIT license to these texts. Original source terms must be checked before any
future redistribution. This study inherits published classifications, without
a new human label audit.

## Gaskell and Bowern gibberish data

Repository: https://github.com/danielgaskell/voynich
Required citation: Gaskell, Daniel E., and Claire L. Bowern (2022).
Gibberish after all? Voynichese is statistically similar to human-produced
samples of meaningless text. International Conference on the Voynich
Manuscript 2022, CEUR Workshop Proceedings 3313.
https://ceur-ws.org/Vol-3313/paper4.pdf

The following notice applies to the upstream code and gibberish data,
explicitly excluding the meaningful and Voynichese collections:

Copyright (c) 2022, Daniel E. Gaskell and Claire L. Bowern.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software, datasets, and associated documentation files (the "Software
and Datasets"), to deal in the Software and Datasets without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software and Datasets, and to
permit persons to whom the Software is furnished to do so, subject to the
following conditions:

- The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software and Datasets.
- Any publications making use of the Software and Datasets, or any substantial
  portions thereof, shall cite the Software and Datasets's original publication:

> Gaskell, Daniel E., Claire L. Bowern, 2022. Gibberish after all? Voynichese
  is statistically similar to human-produced samples of meaningless text. CEUR
  Workshop Proceedings, International Conference on the Voynich Manuscript 2022,
  University of Malta.
  
THE SOFTWARE AND DATASETS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE AND DATASETS.

## Character bigram baseline

`detectors.py` adapts the normalization, smoothed transition counting and
mean-log-likelihood approach from Rob Renaud’s Gibberish-Detector. Training
data, score orientation and threshold selection differ as documented in
`protocol.md`. The trigram extension is study code. Upstream license:

The MIT License (MIT)

Copyright (c) 2015 Rob Renaud

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
