import math
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(n: int, rng: np.random.Generator) -> float:
    # Uniform points in [0,1]x[0,1]
    x = rng.random(n)
    y = rng.random(n)
    inside = (x * x + y * y) <= 1.0
    return 4.0 * inside.mean()


def main() -> None:
    ns = [1_000, 10_000, 100_000, 1_000_000]
    seed = 42
    rng = np.random.default_rng(seed)

    out_dir = Path(__file__).resolve().parent
    results_path = out_dir / "results.csv"
    plot_path = out_dir / "error_vs_n.png"

    rows = []
    for n in ns:
        pi_hat = estimate_pi(n, rng)
        abs_err = abs(pi_hat - math.pi)
        rows.append((n, pi_hat, abs_err))

    # Write CSV
    with results_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "pi_hat", "abs_error"])
        w.writerows(rows)

    # Plot error vs n (log-log)
    n_vals = [r[0] for r in rows]
    err_vals = [r[2] for r in rows]

    plt.figure()
    plt.loglog(n_vals, err_vals, marker="o")
    plt.xlabel("n")
    plt.ylabel("absolute error |π̂ − π|")
    plt.title("Monte Carlo π: error vs n")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)

    print(f"Wrote {results_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
