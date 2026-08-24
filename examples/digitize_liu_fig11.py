"""
Extract the curves of Liu et al. (2024) Figure 11 from a raster of the published figure.

Reference
---------
Z. Liu, G. Wang, X. Rui, G. Wu, J. Tang, L. Gu, "Modeling and simulation framework for
missile launch dynamics in a rigid-flexible multibody system with slider-guide
clearance", Nonlinear Dynamics 112:21701-21728 (2024), Fig. 11 (p. 21717).

Why this exists
---------------
Section 6.3 of the paper validates the method against MSC.ADAMS, but the underlying data
is not published -- the authors state that it cannot be released. The only available form
of the reference result is the printed figure. This script recovers it by color-keyed
pixel extraction so that the comparison in `liu_6_3_validation.py` is reproducible and
its provenance is explicit, rather than being points read off by eye.

Method
------
Each of the three panels is calibrated from its own axis tick marks, located by finding
the spines and the short dark marks beside them. Samples are then taken one image column
at a time: the pixels matching a curve's color are averaged to a single row, which
becomes one (t, value) pair. The two series are separated purely by hue -- solid blue for
MSC.ADAMS, dashed orange for the paper's own method.

Accuracy and its limits
-----------------------
This is a measurement of a printed figure, not the authors' data, and it should be
labeled as such wherever it is plotted. At the resolution available each panel is under
100 pixels tall, so one pixel is worth roughly:

    panel (a)  x_R          0.034 m
    panel (b)  y_ddot_L     0.48 m/s^2
    panel (c)  gamma_dot    0.235 deg/s

With a line roughly 1.5 px wide, a realistic uncertainty is about twice those figures.
Two further effects are not corrected for and are worth remembering when reading the
overlay:

* Where the curves cross, the upper one hides the lower, so samples are missing rather
  than wrong. The dashed orange series is sparse by construction for the same reason.
* Anti-aliased edge pixels are excluded by the color thresholds, which biases each
  sample very slightly toward the line's core -- harmless here, but it means the result
  should never be quoted to more than two significant figures.

Usage
-----
    python examples/digitize_liu_fig11.py <figure_image.png> [-o output.csv]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    raise SystemExit("This script needs Pillow: pip install pillow")


@dataclass(frozen=True)
class PanelCalibration:
    """
    Pixel-to-data mapping for one panel, taken from its axis ticks.

    Attributes
    ----------
    name : str
        Panel label, "a", "b" or "c".
    quantity : str
        What the ordinate is.
    unit : str
        Its unit.
    x_px : tuple of float
        Pixel columns of two known abscissa ticks.
    x_val : tuple of float
        The values those ticks carry [s].
    y_px : tuple of float
        Pixel rows of two known ordinate ticks.
    y_val : tuple of float
        The values those ticks carry.
    col_range : tuple of int
        Columns to search for curve pixels.
    """
    name: str
    quantity: str
    unit: str
    x_px: Tuple[float, float]
    x_val: Tuple[float, float]
    y_px: Tuple[float, float]
    y_val: Tuple[float, float]
    col_range: Tuple[int, int]

    def to_t(self, px: np.ndarray) -> np.ndarray:
        """Map image columns to time [s]."""
        (p0, p1), (v0, v1) = self.x_px, self.x_val
        return v0 + (px - p0) * (v1 - v0) / (p1 - p0)

    def to_value(self, py: np.ndarray) -> np.ndarray:
        """Map image rows to the panel's ordinate."""
        (p0, p1), (v0, v1) = self.y_px, self.y_val
        return v0 + (py - p0) * (v1 - v0) / (p1 - p0)

    @property
    def px_resolution(self) -> float:
        """Ordinate units per pixel row -- the floor on this panel's accuracy."""
        (p0, p1), (v0, v1) = self.y_px, self.y_val
        return abs((v1 - v0) / (p1 - p0))


# Calibration measured from the tick marks of the published figure. The abscissa ticks
# (0.0 and 0.3 s) are 128 px apart in all three panels, which is a useful consistency
# check that the panels were not scaled independently.
PANELS: Tuple[PanelCalibration, ...] = (
    PanelCalibration("a", "x_R", "m",
                     x_px=(65.5, 193.5), x_val=(0.0, 0.3),
                     y_px=(380.5, 293.0), y_val=(0.0, 3.0),
                     col_range=(60, 210)),
    PanelCalibration("b", "y_ddot_L", "m/s^2",
                     x_px=(268.0, 396.0), x_val=(0.0, 0.3),
                     y_px=(352.0, 310.0), y_val=(0.0, 20.0),
                     col_range=(262, 412)),
    PanelCalibration("c", "gamma_dot", "deg/s",
                     x_px=(475.5, 603.5), x_val=(0.0, 0.3),
                     y_px=(316.5, 359.0), y_val=(0.0, -10.0),
                     col_range=(469, 619)),
)

# Row band containing the three plot boxes.
ROW_BAND = (288, 384)

SERIES = {
    "adams": "MSC.ADAMS (solid blue)",
    "liu": "Liu et al. method (dashed orange)",
}


def _masks(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Boolean masks selecting each curve by hue.

    Parameters
    ----------
    img : np.ndarray
        RGB image array.

    Returns
    -------
    dict
        Mask per series key. Thresholds are deliberately strict so that anti-aliased
        edge pixels and the black axes are excluded.
    """
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    return {
        "adams": (b > 110) & (b - r > 45) & (b - g > 25),
        "liu": (r > 140) & (r - b > 55) & (r - g > 25),
    }


def extract(image_path: str) -> List[dict]:
    """
    Digitize every panel and series of the figure.

    Parameters
    ----------
    image_path : str
        Path to the figure raster.

    Returns
    -------
    List[dict]
        One record per sample, with panel, series, time and value.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    masks = _masks(img)
    r0, r1 = ROW_BAND

    rows: List[dict] = []
    for panel in PANELS:
        c0, c1 = panel.col_range
        for series, mask in masks.items():
            for col in range(c0, c1):
                hits = np.where(mask[r0:r1, col])[0]
                if hits.size == 0:
                    continue
                # A column through a steep excursion clips the curve twice -- once on the
                # way up, once on the way down. Averaging the whole column would place a
                # sample in the empty space between the two branches and quietly shave
                # every peak off an oscillatory trace, so each contiguous run is emitted
                # as its own sample instead.
                for run in np.split(hits, np.where(np.diff(hits) > 2)[0] + 1):
                    py = float(run.mean()) + r0
                    rows.append({
                        "panel": panel.name,
                        "quantity": panel.quantity,
                        "unit": panel.unit,
                        "series": series,
                        "t_s": round(float(panel.to_t(np.array(col))), 6),
                        "value": round(float(panel.to_value(np.array(py))), 6),
                    })
    return rows


def main() -> None:
    """Run the extraction and write a CSV with its provenance in the header."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="raster of Liu et al. (2024) Fig. 11")
    ap.add_argument("-o", "--output", default="examples/data/liu2024_fig11_digitized.csv")
    args = ap.parse_args()

    rows = extract(args.image)

    with open(args.output, "w", newline="") as fh:
        fh.write("# Digitized from Liu et al. (2024), Nonlinear Dynamics 112:21701-21728,\n")
        fh.write("# Fig. 11 (p. 21717). NOT the authors' data: these values were recovered\n")
        fh.write("# from the published raster by color-keyed pixel extraction and carry a\n")
        fh.write("# reading uncertainty of roughly two pixels. Per-panel pixel resolution:\n")
        for p in PANELS:
            fh.write(f"#   panel {p.name} ({p.quantity}): {p.px_resolution:.4g} {p.unit} / px\n")
        fh.write("# Quote to no more than two significant figures.\n")
        writer = csv.DictWriter(
            fh, fieldnames=["panel", "quantity", "unit", "series", "t_s", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to {args.output}")
    for p in PANELS:
        for s in SERIES:
            n = sum(1 for r in rows if r["panel"] == p.name and r["series"] == s)
            vals = [r["value"] for r in rows if r["panel"] == p.name and r["series"] == s]
            span = f"{min(vals):+.3g} .. {max(vals):+.3g}" if vals else "-"
            print(f"  panel {p.name} {s:6s}: {n:4d} samples, {p.quantity} in [{span}] {p.unit}")


if __name__ == "__main__":
    main()
