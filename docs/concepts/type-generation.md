# Type generation and SMIRKS specificity

OpenFF parameters are matched to atoms by SMIRKS patterns. Whether a given parameter is shared across an entire family of molecules or is specific to one ring depends on how much detail about the local environment the SMIRKS pattern contains. `presto` generates bespoke SMIRKS for your molecules — this page is about the options that control how specific they are.

For the recipe-level treatment, see **[How-to → Fit a congeneric series](../how-to/fit-congeneric-series.md)**.

## Bespoke vs transferable SMIRKS

Compare two SMIRKS for a C–C bond:

- General: `[#6X4:1]-[#6X4:2]` — any tetrahedral-carbon to tetrahedral-carbon bond.
- Very sppecific: `[#6X4H3:1]-[#6X4H3:1]` -- the carbon-carbon bond in ethane.

OpenFF's standard force fields use mostly very general SMIRKS, which sacrifices accuracy for chemical generality. `presto` flips this trade-off: bespoke SMIRKS are derived from the actual atoms in your molecule, sacrificing generality for accuracy.

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

## Removing terminal caps from generated types

`remove_atom_smarts` removes selected terminal fragments after PRESTO has identified
the ordinary valence terms and chemical context. In each removal SMARTS, mapped atoms
are removed; unmapped atoms provide optional recognition context. Every cut bond is
terminated by an untagged wildcard with the same bond order. The wildcard requires an
attachment without encoding the neighbouring atom's element or chemistry.

For an ACE–amino-acid–NME input, representative masks are:

```yaml
param_settings:
    type_generation_settings:
        Bonds:
            max_extend_distance: -1
            remove_atom_smarts:
                - "[CH3:1][C:2](=[O:3])N[C]"  # mapped ACE atoms are removed
                - "C(=O)[NH:1][CH3:2]"        # mapped NME atoms are removed
```

The setting is per handler, so repeat the masks under `Angles`, `ProperTorsions`, and
`ImproperTorsions` when those types should use the same residue boundary. The default
is an empty list, which preserves the existing type-generation behaviour.

Each mask must match at least once on every input molecule and each distinct selected
fragment must be connected to the retained molecule by exactly one bond. Equivalent
SMARTS embeddings are deduplicated by the atoms selected for removal; genuinely
repeated caps are all removed. Directly attached hydrogens are removed with a selected
heavy atom. PRESTO logs the raw and distinct match counts.

If any force-defining atoms of a bond, angle, or torsion lie in a selected cap, PRESTO
does not create a bespoke type for that term. The original force-field parameter must
remain assigned; generation raises an error if another bespoke pattern would override
it. A masked pattern that collapses parameters from different source types also raises
rather than choosing one source arbitrarily.

Removal masks are chemical queries, not residue labels. A poorly chosen SMARTS can
legitimately select a pendant side chain that also has one attachment bond. Use enough
unmapped context to distinguish the intended cap and inspect the logged match counts.
`max_extend_distance` remains authoritative: caps outside the generated subgraph do
not cause extra wildcard context to be added.

## Why we exclude linear torsions by default

The default `ProperTorsions.exclude` list contains three SMARTS:

```
[*:1]-[*:2]#[*:3]-[*:4]       # triple bond in the middle
[*:1]~[*:2]-[*:3]#[*:4]       # triple bond at the end
[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]   # cumulated double bonds
```

These are torsions across linear (sp-hybridised or cumulated) systems where the dihedral is geometrically ill-defined. Fitting these results in small but non-zero force constants which can produce instabilities during MD.

## Sharing parameters across a congeneric series

For a series of related molecules, set `max_extend_distance` to a finite value for each valence type. We've found that 2 is a reasonable default which allows us to get very similar validation losses for TYK2 ligands compared to completely bespoke types. However, longer ranged patterns will very likely be required for some systems. SMIRKS for substructures shared up to that depth will collapse onto a single parameter, which is then fitted against the combined dataset.

This is intended to reduce noise: chemically equivalent parameters in different molecules can have different fitted values due to per-molecule sampling variance from finite time MD, and sharing forces them to a single consensus value.

For the concrete YAML recipe, see **[Fit a congeneric series](../how-to/fit-congeneric-series.md)**.

## API reference

[`TypeGenerationSettings`](../reference/api/settings.md#presto.settings.TypeGenerationSettings) and the implementation in [`presto.create_types`](../reference/api/create_types.md).
