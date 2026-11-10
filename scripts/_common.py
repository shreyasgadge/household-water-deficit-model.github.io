"""Shared CLI plumbing for the figure scripts."""
import argparse
from pathlib import Path

from supplymodel import load_cluster_parameters
from supplymodel.data import DEFAULT_CSV
from supplymodel.plotting import use_headless_backend

ROOT = Path(__file__).resolve().parents[1]


def parse_args(description, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--params", default=str(DEFAULT_CSV), help="cluster parameter CSV")
    ap.add_argument("--out", default=str(ROOT / "results"), help="output directory")
    ap.add_argument("--show", action="store_true", help="open figure windows instead of headless saving")
    if extra:
        extra(ap)
    args = ap.parse_args()
    if not args.show:
        use_headless_backend()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return args, load_cluster_parameters(args.params)
