"""Export the existing chart from the saved notebook without executing cells."""

import base64
import json
from pathlib import Path


def main() -> None:
    """Copy a real saved notebook figure into the README assets directory."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/04_backtest.ipynb").read_text())
    figures = [
        output["data"]["image/png"]
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    image = figures[0]
    if isinstance(image, list):
        image = "".join(image)
    (root / "assets").mkdir(exist_ok=True)
    (root / "assets/results.png").write_bytes(base64.b64decode(image))


if __name__ == "__main__":
    main()
