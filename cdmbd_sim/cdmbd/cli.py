"""
cdmbd.cli
---------
Command-line entry point.  Installed as ``cdmbd-run`` by pyproject.toml.

Usage
-----
    cdmbd-run              # run all experiments
    cdmbd-run --p1         # P1 only
    cdmbd-run --p2         # P2 only
    cdmbd-run --p3         # P3 only
    cdmbd-run --loop       # design loop only
    cdmbd-run --out ./figs # set output directory
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="cdmbd-run",
        description="C-DMBD paper experiments (Possati 2026)",
    )
    parser.add_argument("--p1",   action="store_true", help="Run P1 – intra-family navigation")
    parser.add_argument("--p2",   action="store_true", help="Run P2 – family transition")
    parser.add_argument("--p3",   action="store_true", help="Run P3 – ontological disambiguation")
    parser.add_argument("--loop", action="store_true", help="Run closed design loop")
    parser.add_argument("--out",  default="figures",   help="Output directory for figures")
    args = parser.parse_args()

    run_all = not any([args.p1, args.p2, args.p3, args.loop])

    # Patch figure output directory
    import cdmbd.phenomena as phen
    import cdmbd.design_loop as dl
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    phen.FIG_DIR = out_dir
    dl.FIG_DIR   = out_dir

    print("=" * 60)
    print("C-DMBD – Design, Cups, and Blankets  (Possati 2026)")
    print("=" * 60)

    t0 = time.time()

    if run_all or args.p1:
        t = time.time()
        phen.demo_p1()
        print(f"  [P1 completed in {time.time()-t:.1f}s]")

    if run_all or args.p2:
        t = time.time()
        phen.demo_p2()
        print(f"  [P2 completed in {time.time()-t:.1f}s]")

    if run_all or args.p3:
        t = time.time()
        phen.demo_p3()
        print(f"  [P3 completed in {time.time()-t:.1f}s]")

    if run_all or args.loop:
        t = time.time()
        dl.demo_design_loop()
        print(f"  [Loop completed in {time.time()-t:.1f}s]")

    print(f"\nAll done in {time.time()-t0:.1f}s  →  figures in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
