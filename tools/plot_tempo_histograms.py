#!/usr/bin/env python3
"""Plot tempo histograms for all datasets in the data/ directory."""

from pathlib import Path

import matplotlib.pyplot as plt
import mirdata

import musicality.dataformats as dataformats

DATA_DIR = Path(__file__).parent.parent / dataformats.load().data_dir


def load_tempos(name: str, path: Path) -> list[float]:
    ds = mirdata.initialize(name, data_home=str(path))
    tempos = []
    for tid in ds.track_ids:
        t = ds.track(tid).tempo
        if t is not None:
            tempos.append(t)
    return tempos


def main():
    available = mirdata.list_datasets()
    datasets = sorted(e.name for e in DATA_DIR.iterdir() if e.is_dir() and e.name in available)

    if not datasets:
        print("No recognised mirdata datasets found in data/.")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, name in zip(axes, datasets):
        path = DATA_DIR / name
        tempos = load_tempos(name, path)

        if not tempos:
            ax.set_title(f"{name}\n(no tempo annotations)")
            continue

        ax.hist(tempos, bins=30, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{name}  (n={len(tempos)})")
        ax.set_xlabel("Tempo (BPM)")
        ax.set_ylabel("Count")
        ax.axvline(sum(tempos) / len(tempos), color="red", linestyle="--", linewidth=1, label=f"mean {sum(tempos)/len(tempos):.0f}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path("tempo_histograms.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
