# Third-party software and data

Written by `scripts/third_party.py` from the metadata of the packages in `requirements/*.txt`; do not edit by hand.
A package is listed with the licence its own metadata declares.

## Models and data

| What | Licence | Source | How it gets here |
|---|---|---|---|
| PP-OCRv6 small, detection and recognition (ONNX) | Apache-2.0 | https://github.com/RapidAI/RapidOCR (converted from PaddleOCR, https://github.com/PaddlePaddle/PaddleOCR) | downloaded by scripts/fetch_models.py, not stored in this repository |
| Word lists, spelling index and character models built from wordfreq data | CC BY-SA 4.0 | https://github.com/rspeer/wordfreq | built on install by scripts/build_dictionaries.py; derived files keep the licence, see MODELS.md |

## Python packages

| Package | Versions | Licence | Home page |
|---|---|---|---|
| antlr4-python3-runtime | 4.9.3 | BSD | http://www.antlr.org |
| attrs | 26.1.0 | MIT | https://github.com/python-attrs/attrs |
| certifi | 2026.7.22 | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | MIT | https://pypi.org/project/charset-normalizer/ |
| coloredlogs | 15.0.1 | MIT | https://coloredlogs.readthedocs.io |
| colorlog | 6.12.0 | MIT License | https://github.com/borntyping/python-colorlog |
| et-xmlfile | 2.0.0 | MIT | https://foss.heptapod.net/openpyxl/et_xmlfile |
| exceptiongroup | 1.3.1 | MIT License | https://github.com/agronholm/exceptiongroup |
| flatbuffers | 25.12.19 | Apache 2.0 | https://google.github.io/flatbuffers/ |
| ftfy | 6.3.1 | Apache-2.0 | https://ftfy.readthedocs.io/en/latest/ |
| humanfriendly | 10.0 | MIT | https://humanfriendly.readthedocs.io |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna |
| iniconfig | 2.3.0 | MIT | https://github.com/pytest-dev/iniconfig |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema |
| jsonschema-specifications | 2025.9.1 | MIT | https://github.com/python-jsonschema/jsonschema-specifications |
| langcodes | 3.5.1 | MIT License | https://github.com/georgkrause/langcodes |
| lightgbm | 4.7.0 | MIT | https://github.com/lightgbm-org/LightGBM |
| locate | 1.1.1 | MIT License | https://github.com/AutoActuary/locate |
| lxml | 6.1.3 | BSD-3-Clause | https://lxml.de/ |
| mpmath | 1.3.0 | BSD | http://mpmath.org/ |
| msgpack | 1.2.3 | Apache-2.0 | https://msgpack.org/ |
| narwhals | 2.26.0 | MIT | https://github.com/narwhals-dev/narwhals |
| numpy | 2.2.6 | BSD License | https://numpy.org |
| omegaconf | 2.3.1 | BSD License | https://github.com/omry/omegaconf |
| onnxruntime | 1.23.2 | MIT License | https://onnxruntime.ai |
| opencv-python | 5.0.0.93 | Apache 2.0 | https://github.com/opencv/opencv-python |
| openpyxl | 3.1.5 | MIT | https://openpyxl.readthedocs.io |
| openvino | 2026.4.0 | Apache-2.0 | https://docs.openvino.ai/2023.0/index.html |
| openvino-telemetry | 2025.2.0 | OSI Approved :: Apache Software License | https://github.com/openvinotoolkit/telemetry |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pikepdf | 10.13.0.post1 | MPL-2.0 | https://github.com/pikepdf/pikepdf |
| pillow | 12.3.0 | MIT-CMU | https://python-pillow.github.io |
| pip | 26.2.1 | MIT | https://pip.pypa.io/ |
| pluggy | 1.6.0 | MIT | https://pypi.org/project/pluggy/ |
| protobuf | 7.36.2 | 3-Clause BSD License | https://developers.google.com/protocol-buffers/ |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| pyclipper | 1.4.0 | MIT | https://github.com/fonttools/pyclipper |
| pygments | 2.21.0 | BSD-2-Clause | https://pygments.org |
| pypdfium2 | 5.13.0 | BSD-3-Clause, Apache-2.0, dependency licenses | https://github.com/pypdfium2-team/pypdfium2 |
| pytest | 9.1.1 | MIT | https://docs.pytest.org/en/latest/ |
| pyyaml | 6.0.3 | MIT | https://pyyaml.org/ |
| rapidfuzz | 3.14.5 | MIT | https://github.com/rapidfuzz/RapidFuzz |
| rapidocr | 3.9.2 | Apache-2.0 | https://pypi.org/project/rapidocr/ |
| referencing | 0.37.0 | MIT | https://github.com/python-jsonschema/referencing |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | https://github.com/mrabarnett/mrab-regex |
| reportlab | 3.6.13, 5.0.1 | BSD License | http://www.reportlab.com/ |
| requests | 2.34.2 | Apache-2.0 | https://github.com/psf/requests |
| rpds-py | 0.30.0, 2026.6.3 | MIT | https://github.com/crate-py/rpds |
| ruff | 0.16.10 | MIT | https://docs.astral.sh/ruff |
| scipy | 1.15.3 | BSD License | https://scipy.org/ |
| setuptools | 84.0.0 | MIT | https://github.com/pypa/setuptools |
| shapely | 2.1.2 | BSD 3-Clause | https://github.com/shapely/shapely |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six |
| sympy | 1.14.0 | BSD | https://sympy.org |
| tomli | 2.4.1 | MIT | https://github.com/hukkin/tomli |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | https://tqdm.github.io |
| typing-extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| urllib3 | 2.8.0 | MIT | https://pypi.org/project/urllib3/ |
| wcwidth | 0.9.1 | MIT License | https://github.com/jquast/wcwidth |
| wheel | 0.48.0 | MIT | https://github.com/pypa/wheel |
| wordfreq | 3.1.1 | Apache-2.0 | https://github.com/rspeer/wordfreq/ |

## Usage reporting of a dependency

OpenVINO sends an anonymous usage event to Google Analytics each time it is imported, unless a consent file in the home folder says no or the process runs with `CI=true`. Extract imports it with `CI=true` (`src/extract/engine/quiet_openvino.py`, tested); see B45 in docs/release-notes.md.
