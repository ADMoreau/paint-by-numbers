"""A fixed set of paints for algorithm tests, so they don't depend on what the database holds.

K and S come from each paint's published color and transparency rating alone (fit.estimate_ks),
not from the stored fit, so refitting the database (as the tint and glaze photos did) doesn't move
these tests.
"""

import dataclasses

from paint_by_numbers import fit, paints

CORE_NAMES = (
    "Burnt Umber",
    "Cadmium Yellow Primrose",
    "Carbon Black",
    "Green Gold",
    "Phthalo Blue (Green Shade)",
    "Pyrrole Red",
    "Quinacridone Magenta",
    "Titanium White",
    "Ultramarine Blue",
)


def core_paints() -> list[paints.Paint]:
    by_name = {p.name: p for p in paints.load_paints()}
    out = []
    for name in CORE_NAMES:
        p = by_name[name]
        K, S = fit.estimate_ks(p.lab, p.transparency)
        out.append(dataclasses.replace(p, K=tuple(fit._rounded(K)), S=tuple(fit._rounded(S))))
    return out


def spectrum(linear_rgb) -> "torch.Tensor":
    """A working-space (spectral) reflectance with the color of ``linear_rgb`` (linear sRGB), for
    test backgrounds written as three numbers."""
    import torch

    from paint_by_numbers import color

    return color.xyz_to_working(color.linear_to_xyz(torch.as_tensor(linear_rgb, dtype=torch.float64)))
