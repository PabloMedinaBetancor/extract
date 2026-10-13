"""Every script in `examples/` runs to the end on the real models, in a folder that holds the sample files.

Runs only with `EXTRACT_DOCS=1` (`scripts/docs.py check` sets it). A script that is not meant to be run alone says so
on its first line with `# docs-test: skip - the reason`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests import docs_support as ds

pytestmark = [
    pytest.mark.docs,
    pytest.mark.skipif(not ds.enabled(), reason="real-model examples: set EXTRACT_DOCS=1 or run scripts/docs.py check"),
]

SCRIPTS = sorted((ds.ROOT / "examples").glob("*.py")) if (ds.ROOT / "examples").is_dir() else []


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_example_script_runs(script):
    first = script.read_text(encoding="utf-8").splitlines()[:1]
    skip = re.match(r"#\s*docs-test:\s*skip\s*-\s*(.+)", first[0]) if first else None
    if skip:
        pytest.skip(skip.group(1))
    work = Path(tempfile.mkdtemp(prefix="extract-example-"))
    try:
        shutil.copytree(ds.samples(), work / "cwd")
        shutil.copy2(script, work / "cwd" / script.name)
        done = subprocess.run(
            [sys.executable, script.name],
            cwd=work / "cwd",
            env=ds.child_env(),
            capture_output=True,
            text=True,
            timeout=ds.REAL_MODEL_TIMEOUT,
        )
    finally:
        shutil.rmtree(work, True)
    status = "ok" if done.returncode == 0 else "fail"
    report = os.environ.get("EXTRACT_DOCS_REPORT")
    if report:
        entry = {
            "page": f"examples/{script.name}",
            "results": [{"number": 1, "line": 1, "status": status, "error": done.stderr[-300:] or None}],
        }
        with open(report, "a", encoding="utf-8") as out:
            out.write(json.dumps(entry) + "\n")
    assert done.returncode == 0, f"examples/{script.name}: exit {done.returncode}\n{done.stderr[-1500:]}"
