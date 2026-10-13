# Licence and credits

What is licensed under what, and where each credit lives. Everything Extract loads that it does not own keeps its own
licence, and this page names each one.

## Extract

Extract is released under the MIT licence. The text and the copyright line are in the file `LICENSE` at the root of the
repository. The two guard models, `guard.model` and `guard-thorough.model`, were trained for Extract by its author with LightGBM (MIT)
and carry the same licence as the repository. Each is an archive of decision-tree files and a table of numbers
(thresholds and calibration): it holds no word of the word lists below and no text of any benchmark document. The sample documents in `tests/fixtures` are invented: every name, address and
identifier in them is made up and cannot exist.

## The OCR models

The OCR models are PP-OCR models of the PaddleOCR project, converted to ONNX and published by the RapidOCR project. All five
are under the Apache-2.0 licence. They are not stored in the repository: `scripts/fetch_models.py` downloads them from
the address in `models/models.json` and checks each file against the SHA-256 recorded there. Sizes are in megabytes of
10^6 bytes, from the same file.

| Model | Size | Role | Downloaded by default |
|---|---|---|---|
| `PP-OCRv6_det_small` | 9.9 MB | Finds the text lines on the page. | Yes |
| `PP-OCRv6_rec_small` | 21.2 MB | Reads the text of each line. | Yes |
| `PP-OCRv6_rec_medium` | 76.6 MB | The `deep` reader of the thorough second reading. | No: `--all` |
| `latin_PP-OCRv5_rec_mobile` | 7.9 MB | The `latin` reader of the second reading. | No: `--all` |
| `ch_PP-OCRv5_rec_server` | 84.6 MB | A candidate in the comparison that chose the readers. It lost, and Extract never loads it. | No: `--all` |

The two default models are 31 MB together. The three others are 169.1 MB, and the thorough mode uses two of them (84.5 MB).

## The word lists

The word lists the guard compares words with are built on install, by `scripts/build_dictionaries.py`, from the wordfreq
package. wordfreq's data is licensed CC BY-SA 4.0, and the lists derived from it keep that licence. The credit, in the words of
`MODELS.md`:

> The word lists are derived (by scripts/build_dictionaries.py) from wordfreq by Robyn Speer
> (https://github.com/rspeer/wordfreq), whose data is licensed CC BY-SA 4.0
> (https://creativecommons.org/licenses/by-sa/4.0/); the derived files keep that licence. wordfreq's data includes Google
> Books Ngrams (http://books.google.com/ngrams), the Leeds Internet Corpus, Wikipedia, ParaCrawl, OPUS OpenSubtitles 2018
> (data from OpenSubtitles, http://www.opensubtitles.org/) and the SUBTLEX word lists created by Marc Brysbaert et al.,
> which are freely available data (http://crr.ugent.be/programs-data/subtitle-frequencies).

The files are `dict_es.tsv.gz` and `dict_en.tsv.gz` (the 100,000 most frequent words of each language, with about 100
hand-written invoice and administrative words), `symdel_es.npy` and `symdel_en.npy` (a spelling index) and `charlm_es.npy`
and `charlm_en.npy` (character statistics). They are not in the git repository, and `scripts/build_dictionaries.py`
rebuilds them byte for byte. `MODELS.md` lists every data file with its licence, source and size. No text of any benchmark
document is in these files.

## Python packages

Extract uses open-source Python packages, each under its own licence. `THIRD-PARTY.md`, at the root of the repository, lists
every package of the lock files with its version, the licence its own metadata declares and its home page. It is written by
`scripts/third_party.py`, so it does not drift from what is installed. Most are MIT, BSD or Apache-2.0; read the file for
the rest.

One package, `antlr4-python3-runtime` 4.9.3, is offered on PyPI only as a source archive. The wheel in `requirements/wheels/` was built from that archive of that exact release,
and it is shipped with the BSD 3-clause licence text of the ANTLR project, `requirements/wheels/antlr4-python3-runtime-4.9.3.LICENSE.txt`. The second notice in that file, for two
JavaScript files, belongs to the JavaScript runtime of ANTLR and not to this package.

The documentation site is built with MkDocs and its Material theme. They are installed only to build the site and are not
part of Extract.

## Font metrics

The invisible text layer of the searchable PDF is placed with the widths of the standard Helvetica font, taken from Adobe's
AFM font metrics. The table is in `src/extract/outputs/helvetica.py`.

## Support

Questions and bug reports: open an issue on GitHub. Work requests go through the author's Upwork profile.

