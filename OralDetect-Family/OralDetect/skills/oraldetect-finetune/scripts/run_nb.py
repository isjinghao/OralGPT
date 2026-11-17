#!/usr/bin/env python
"""Execute a notebook in place, without depending on a `jupyter` CLI.

    python run_nb.py <notebook.ipynb> [--timeout 1800] [--to-html]

`jupyter nbconvert --execute` is the documented incantation everywhere, and it is the one thing
about this skill that reliably does not exist: the console is often a conda env where nbconvert is
importable but no `jupyter` entry point is on PATH, and where the host's own Jupyter config (Baidu
AIStudio's, for one) is loaded on the way in and buries the real output in warnings.

So this runs the notebook through the Python API instead, and pins two things down:

* **the kernel is this interpreter.** A registered `python3` kernelspec can point anywhere; the
  notebook has to run where matplotlib and PIL actually are, which is here. A throwaway kernelspec
  for `sys.executable` is written to a temp dir and `JUPYTER_PATH` is pointed at it alone.
* **no host config is read.** `JUPYTER_NO_CONFIG` sends jupyter_core at an empty config dir.

The notebook runs with its own directory as cwd, so figures land beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import tempfile

PIP = "pip install nbclient nbformat ipykernel"


def isolate(tmp: str) -> None:
    """Make this process's Jupyter see exactly one kernel -- ours -- and no user config."""
    kdir = osp.join(tmp, "kernels", "runnb")
    os.makedirs(kdir, exist_ok=True)
    with open(osp.join(kdir, "kernel.json"), "w") as f:
        json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": f"python ({osp.basename(sys.executable)})",
                   "language": "python"}, f)
    os.environ["JUPYTER_PATH"] = tmp
    os.environ["JUPYTER_NO_CONFIG"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")      # no display on a login node or a batch job


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute a notebook in place.")
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    ap.add_argument("--to-html", action="store_true",
                    help="also write <notebook>.html next to it")
    a = ap.parse_args()

    nb_path = osp.abspath(a.notebook)
    if not osp.isfile(nb_path):
        sys.exit(f"FATAL: no such notebook -- {nb_path}")

    try:
        import ipykernel  # noqa: F401
        import nbformat
    except ImportError as e:
        sys.exit(f"FATAL: {e.name} is not installed in {sys.executable}.\n       {PIP}")

    with tempfile.TemporaryDirectory() as tmp:
        isolate(tmp)
        try:
            from nbclient import NotebookClient
        except ImportError:
            try:
                from nbconvert.preprocessors import ExecutePreprocessor
            except ImportError:
                sys.exit(f"FATAL: neither nbclient nor nbconvert is installed in "
                         f"{sys.executable}.\n       {PIP}")
            NotebookClient = None

        nb = nbformat.read(nb_path, as_version=4)
        print(f"executing {nb_path}\n  kernel {sys.executable}", flush=True)
        try:
            if NotebookClient is not None:
                NotebookClient(nb, timeout=a.timeout, kernel_name="runnb",
                               resources={"metadata": {"path": osp.dirname(nb_path)}}).execute()
            else:
                ExecutePreprocessor(timeout=a.timeout, kernel_name="runnb").preprocess(
                    nb, {"metadata": {"path": osp.dirname(nb_path)}})
        except Exception as e:
            # Write what did run before re-raising: a half-executed notebook still shows which
            # cell broke and what it printed, which is the whole diagnosis.
            nbformat.write(nb, nb_path)
            print(f"\nFATAL: the notebook raised -- {type(e).__name__}: {e}", file=sys.stderr)
            print(f"       partial output kept in {nb_path}", file=sys.stderr)
            return 1

        nbformat.write(nb, nb_path)
        print(f"  wrote {nb_path}")

        pngs = [osp.join(osp.dirname(nb_path), f)
                for f in sorted(os.listdir(osp.dirname(nb_path) or "."))
                if f.startswith(osp.splitext(osp.basename(nb_path))[0]) and f.endswith(".png")]
        for p in pngs:
            print(f"  figure {p}  ({osp.getsize(p) / 1e3:.0f} kB)")

        if a.to_html:
            try:
                from nbconvert import HTMLExporter
            except ImportError:
                print("  (--to-html skipped: nbconvert is not installed)")
                return 0
            html, _ = HTMLExporter(exclude_input=False).from_notebook_node(nb)
            out = osp.splitext(nb_path)[0] + ".html"
            with open(out, "w") as f:
                f.write(html)
            print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
