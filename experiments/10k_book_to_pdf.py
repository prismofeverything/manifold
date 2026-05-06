"""Combine the per-entry PNG figures from out/book/ into a single PDF
in canonical order (L1.1a, L1.1b, ..., L3.3d).

Plus a title page and a brief table of contents page at the front.
"""

import os
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BOOK_DIR = "out/book"
OUTPUT = "out/book/manifold_book_of_diagrams.pdf"


ENTRIES_INFO = [
    ("L1.1a", "Adaptation",                "leaky integrator + high-pass output"),
    ("L1.1b", "Tracker",                   "low-pass integrator (the memory primitive)"),
    ("L1.1c", "Pure oscillator",           "Kuramoto, single node, intrinsic ω"),
    ("L1.1d", "Adaptive oscillator",       "amplitude tracks input + intrinsic ω"),
    ("L2.2a", "Mutual excitation",         "in-phase Kuramoto synchronization"),
    ("L2.2b", "Mutual inhibition",         "winner-take-all settling"),
    ("L2.2c", "Rivalry",                   "mutual inhibition + slow adaptation"),
    ("L2.2d", "Plastic Hebbian pair",      "self-strengthening coupling"),
    ("L3.3a", "Feedforward chain",         "cascaded low-pass filtering"),
    ("L3.3b", "Bridge",                    "indirect synchronization through B"),
    ("L3.3c", "Triangle",                  "all-to-all densest sync"),
    ("L3.3d", "Cyclic inhibition",         "3-state winnerless competition"),
]


def make_title_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.66, "Manifold", ha="center", fontsize=48, fontweight="bold")
    fig.text(0.5, 0.58, "Book of Diagrams", ha="center", fontsize=28)
    fig.text(0.5, 0.50, "Canonical networks at Levels 1, 2, 3", ha="center",
             fontsize=14, style="italic")
    fig.text(0.5, 0.36, ("Each entry: a categorical / wiring schematic, the dynamics\n"
                        "equations, parameters, and a behavior plot from a small\n"
                        "simulation of that exact specification."),
             ha="center", fontsize=11)
    fig.text(0.5, 0.20,
             "categorical structure:  graph-as-category within entries,\n"
             "operad-of-wiring-diagrams between entries,\n"
             "coalgebra isomorphism for behavioral equivalence",
             ha="center", fontsize=10, style="italic", color="gray")
    fig.text(0.5, 0.05, "manifold project · 2026-05", ha="center", fontsize=9, color="gray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_toc_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.92, "Contents", ha="center", fontsize=22, fontweight="bold")
    y = 0.82
    for i, (eid, name, desc) in enumerate(ENTRIES_INFO, start=1):
        # Group separator between levels
        if i > 1 and ENTRIES_INFO[i - 1][0][1] != ENTRIES_INFO[i - 2][0][1]:
            y -= 0.025
        page_num = i + 2  # +2 for title page and this TOC page
        fig.text(0.10, y, f"page {page_num:>3}", fontsize=9, color="gray", family="monospace")
        fig.text(0.20, y, eid, fontsize=10, fontweight="bold", family="monospace")
        fig.text(0.32, y, name, fontsize=11)
        fig.text(0.55, y, desc, fontsize=10, color="#444444", style="italic")
        y -= 0.045
    fig.text(0.5, 0.04,
             "Levels are graded by node count; letters distinguish variants within a level.",
             ha="center", fontsize=9, color="gray", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_entry_page(pdf, png_path):
    img = mpimg.imread(png_path)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    pngs = [f for f in os.listdir(BOOK_DIR) if f.endswith(".png")]

    def sort_key(name):
        m = re.match(r"L(\d+)_(\d+)([a-z]+)_", name)
        if not m:
            return (99, 99, "z")
        return (int(m.group(1)), int(m.group(2)), m.group(3))

    pngs.sort(key=sort_key)
    print(f"Combining {len(pngs)} entries into PDF...")

    with PdfPages(OUTPUT) as pdf:
        make_title_page(pdf)
        make_toc_page(pdf)
        for p in pngs:
            print(f"  {p}")
            make_entry_page(pdf, os.path.join(BOOK_DIR, p))
        # Set PDF metadata
        d = pdf.infodict()
        d["Title"] = "Manifold — Book of Diagrams (Levels 1–3)"
        d["Subject"] = "Canonical 1-, 2-, 3-node networks with schematic, equations, and behavior"

    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"\nWrote {OUTPUT} ({size_mb:.1f} MB, {len(pngs) + 2} pages)")


if __name__ == "__main__":
    main()
