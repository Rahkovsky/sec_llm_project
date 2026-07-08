#!/usr/bin/env python3
"""Re-render dag.{html,dot,png,pdf} from existing dag.json (no models needed).

    python tools/provenance_html.py [--out out/provenance]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.provenance import ProvenanceDAG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/provenance")
    args = ap.parse_args()

    dot = shutil.which("dot")
    written = 0
    for jf in sorted(Path(args.out).glob("*/dag.json")):
        dag = ProvenanceDAG.from_dict(json.loads(jf.read_text()))
        d = jf.parent
        (d / "dag.html").write_text(dag.to_html(d.name))
        (d / "dag.dot").write_text(dag.to_dot())
        if dot:
            subprocess.run([dot, "-Tpng", "-Gdpi=170", str(d / "dag.dot"),
                            "-o", str(d / "dag.png")], check=False)
            subprocess.run([dot, "-Tpdf", str(d / "dag.dot"),
                            "-o", str(d / "dag.pdf")], check=False)
        written += 1
    print(f"re-rendered {written} DAGs (html/dot/png/pdf) under {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
