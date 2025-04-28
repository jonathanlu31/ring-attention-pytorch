"""
Compare two output files produced by run_single_model.py.

Example:
python compare_outputs.py \
    --file-a /tmp/ring_out.pt \
    --file-b /tmp/normal_out.pt \
    --atol 1e-2
"""
from pathlib import Path

import click
import torch


@click.command()
@click.option("--file-a", required=True, help="Path to first .pt file.")
@click.option("--file-b", required=True, help="Path to second .pt file.")
@click.option("--atol", default=1e-1, help="Absolute tolerance for allclose().")
def main(file_a: str, file_b: str, atol: float) -> None:
    a_path, b_path = Path(file_a), Path(file_b)
    assert a_path.exists() and b_path.exists(), "One or both files are missing."

    print(f"🔍 Loading tensors:\n  A = {a_path}\n  B = {b_path}")
    a = torch.load(a_path, map_location="cpu")["output"]
    b = torch.load(b_path, map_location="cpu")["output"]

    same_shape = a.shape == b.shape
    print(f" • shapes equal? {same_shape} ({a.shape} vs {b.shape})")

    close = torch.allclose(a, b, atol=atol)
    if close and same_shape:
        print(f"✅ Outputs are equal within atol={atol}.")
    else:
        max_diff = (a - b).abs().max().item()
        print(
            f"❌ Outputs differ (max abs diff = {max_diff:.3e}, atol={atol}). "
            "Consider adjusting tolerance or re-running."
        )
        print(a)
        print(b)


if __name__ == "__main__":
    main()
