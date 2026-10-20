"""Shared helpers for building and executing the Phase-2/3 notebooks."""
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent

SETUP = '''import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
FIG, TAB, PRED = ROOT/"results"/"figures", ROOT/"results"/"tables", ROOT/"results"/"predictions"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})
def savefig(name):
    plt.savefig(FIG/f"{name}.png", bbox_inches="tight"); print("saved ->", f"results/figures/{name}.png")
pd.set_option("display.width", 160, "display.max_columns", 40, "display.float_format", lambda v: f"{v:,.3f}")
'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def build(name, cells, timeout=3000):
    nb = new_notebook(cells=[code(SETUP)] + cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / f"{name}.ipynb"
    print(f"--- executing {out.name}", flush=True)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, out)
    print("    wrote", out, flush=True)
