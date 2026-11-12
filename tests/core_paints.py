"""A fixed set of paints for algorithm tests, so they don't depend on what the database holds."""

from paint_by_numbers import paints

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
    return [by_name[n] for n in CORE_NAMES]
