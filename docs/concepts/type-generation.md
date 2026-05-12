# Type generation and SMIRKS specificity

OpenFF parameters are matched to atoms by SMIRKS patterns. Whether a given parameter is shared across an entire family of molecules or is specific to one ring depends on how tightly the SMIRKS is written. `presto` generates bespoke SMIRKS for your molecules — this page is about the knobs that control how specific they are.

For the recipe-level treatment, see **[How-to → Fit a congeneric series](../how-to/fit-congeneric-series.md)**.

## Bespoke vs transferable SMIRKS

Compare two SMIRKS for a C–C bond:

- Transferable: `[#6X4:1]-[#6X4:2]` — any tetrahedral-carbon to tetrahedral-carbon bond.
- Bespoke (specific): `[#6X4H3:1]-[#6X4H2;$(*-[#7])]` — methyl carbon bonded to a methylene that's bonded to a nitrogen.

OpenFF's standard force fields use transferable SMIRKS, which sacrifices accuracy for chemical generality. `presto` flips this trade-off: bespoke SMIRKS are derived from the actual atoms in your molecule, sacrificing generality for fit-time accuracy.

## `max_extend_distance` explained

For each tagged atom in a SMIRKS, `presto` extends the SMARTS pattern outward to include neighbouring atoms. `max_extend_distance` caps how far that extension goes:

```
Tagged bond: A–B

max_extend_distance = 0:   A     – B
max_extend_distance = 1:   X–A   – B–Y      (one bond past each tagged atom)
max_extend_distance = 2:   X–X–A – B–Y–Y    (two bonds past each tagged atom)
max_extend_distance = -1:  ...fully specific (entire molecule)
```

With `-1` (the default), every atom in the molecule is encoded into every SMIRKS. With `2`, the SMIRKS looks two bonds out from the tagged atoms. With `1`, it looks one bond out.

The smaller the number, the more SMIRKS collisions across molecules in a congeneric series — i.e. the more **parameter sharing** you get.

## `include` vs `exclude` (mutually exclusive)

`TypeGenerationSettings` has two list fields that filter which parameters get bespoke types at all:

- `include` — only generate bespoke types for SMIRKS matching this list. Everything else stays at the OpenFF default.
- `exclude` — generate bespoke types for everything *except* SMIRKS matching this list.

You can set one or the other, not both. This is enforced by `TypeGenerationSettings.validate_include_exclude`. The default is empty `include` and empty `exclude`, which means "make everything bespoke".

## Why we exclude linear torsions by default

The default `ProperTorsions.exclude` list contains three SMARTS:

```
[*:1]-[*:2]#[*:3]-[*:4]       # triple bond in the middle
[*:1]~[*:2]-[*:3]#[*:4]       # triple bond at the end
[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]   # cumulated double bonds
```

These are torsions across linear (sp-hybridised or cumulated) systems where the dihedral is geometrically ill-defined. Trying to fit a torsion force constant to them leads to runaway values that destabilise sampling.

These exclusions are kept in sync with the equivalent exclusions in the OpenFF 2.2.x default force field — see `presto/settings.py` for the source.

## Sharing parameters across a congeneric series

For a series of related molecules, set `max_extend_distance: 2` (or smaller) for each valence type. SMIRKS for substructures shared up to that depth will collapse onto a single parameter, which is then fitted against the combined dataset.

This reduces noise: chemically equivalent parameters in different molecules can have different fitted values due to per-molecule sampling variance, and sharing forces them to a single consensus value.

For the concrete YAML recipe, see **[Fit a congeneric series](../how-to/fit-congeneric-series.md)**.

## API reference

[`TypeGenerationSettings`](../reference/api/settings.md#presto.settings.TypeGenerationSettings) and the implementation in [`presto.create_types`](../reference/api/create_types.md).
