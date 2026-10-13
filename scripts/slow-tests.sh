#!/bin/sh
# The long suites kept out of the per-commit run (tests/slow has no __init__.py, so `discover` does not enter it).
# Run before every read of the test split and in the final review.
cd "$(dirname "$0")/.." || exit 2
exec python3 -s -m unittest tests.slow.test_checks_exhaustive tests.slow.test_checks_inject "$@"
