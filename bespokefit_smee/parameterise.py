"""Functionality for generating the initial parameterisation."""

import collections
import copy
import math
from typing import Callable, cast

import loguru
import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
from openff.units import Quantity
from openff.units import unit as off_unit
from rdkit import Chem

from .settings import ParameterisationSettings
from .utils.typing import TorchDevice

logger = loguru.logger

_UNITLESS = off_unit.dimensionless
_ANGSTROM = off_unit.angstrom
_RADIANS = off_unit.radians
_KCAL_PER_MOL = off_unit.kilocalories_per_mole
_KCAL_PER_MOL_ANGSQ = off_unit.kilocalories_per_mole / off_unit.angstrom**2
_KCAL_PER_MOL_RADSQ = off_unit.kilocalories_per_mole / off_unit.radians**2


def _reflect_angle(angle: float) -> float:
    """Reflect an angle (in radians) to be in the range [0, pi)."""
    return math.pi - abs((angle % (2 * math.pi)) - math.pi)


def _add_parameter_with_overwrite(
    handler: openff.toolkit.typing.engines.smirnoff.parameters.ParameterHandler,
    parameter_dict: dict[str, str | Quantity],
) -> None:
    """Add a parameter to a handler, overwriting any existing parameter with the same smirks."""
    old_parameter = handler.get_parameter({"smirks": parameter_dict["smirks"]})
    new_parameter = handler._INFOTYPE(**parameter_dict)
    if old_parameter:
        assert len(old_parameter) == 1
        old_parameter = old_parameter[0]
        logger.info(
            f"Overwriting existing parameter with smirks {parameter_dict['smirks']}."
        )
        idx = handler._index_of_parameter(old_parameter)
        handler._parameters[idx] = new_parameter
    else:
        handler._parameters.append(new_parameter)


def convert_to_smirnoff(
    ff: smee.TensorForceField, base: openff.toolkit.ForceField | None = None
) -> openff.toolkit.ForceField:
    """Convert a tensor force field that *contains bespoke valence parameters* to
    SMIRNOFF format.
    Args:
        ff: The force field containing the bespoke valence terms.
        base: The (optional) original SMIRNOFF force field to add the bespoke
            parameters to. If no specified, a force field containing only the bespoke
            parameters will be returned.
    Returns:
        A SMIRNOFF force field containing the valence terms of the input force field.
    """
    ff_smirnoff = openff.toolkit.ForceField() if base is None else copy.deepcopy(base)

    for potential in ff.potentials:
        if potential.type in {
            "Bonds",
            "Angles",
            "ProperTorsions",
            "ImproperTorsions",
        }:
            assert potential.attribute_cols is None
            parameters_by_smarts: dict[str, dict[int | None, torch.Tensor]] = (
                collections.defaultdict(dict)
            )
            for parameter, parameter_key in zip(
                potential.parameters, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler(potential.type)
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict: dict[str, str | Quantity] = {
                    "smirks": smarts,
                    "id": parameter_id,
                }
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * potential.parameter_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(potential.parameter_cols)
                    }
                )
                _add_parameter_with_overwrite(handler, parameter_dict)

        elif potential.type == "LinearBonds":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                b1 = param[2].item()
                b2 = param[3].item()
                k = k1 + k2
                b = (k1 * b1 + k2 * b2) / k
                dt = param.dtype
                new_params.append([k, b])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL_ANGSQ, _ANGSTROM)
            reconstructed_cols = ("k", "length")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler("Bonds")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                _add_parameter_with_overwrite(handler, parameter_dict)

        elif potential.type == "LinearAngles":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                a1 = param[2].item()
                a2 = param[3].item()
                k = k1 + k2
                # Set k and angle to 0 if very close
                a = (k1 * a1 + k2 * a2) / k
                # Ensure that the angle is in the range [0, pi)
                a = _reflect_angle(a)
                dt = param.dtype
                new_params.append([k, a])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL_RADSQ, _RADIANS)
            reconstructed_cols = ("k", "angle")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler("Angles")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                _add_parameter_with_overwrite(handler, parameter_dict)

        elif potential.type == "LinearProperTorsions":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                periodicity = param[2].item()
                # Params 3 and 4 are phase1 and phase2
                idivf = param[5].item()
                k = k1 + k2
                if k == 0.0:
                    phase = 0.0
                else:
                    phase = math.acos((k1 - k2) / k)
                dt = param.dtype
                new_params.append([k, periodicity, phase, idivf])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_torsion_units = (
                _KCAL_PER_MOL,
                _UNITLESS,
                _RADIANS,
                _UNITLESS,
            )
            reconstructed_torsion_cols = ("k", "periodicity", "phase", "idivf")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler("ProperTorsions")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_torsion_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_torsion_cols)
                    }
                )
                _add_parameter_with_overwrite(handler, parameter_dict)

        elif potential.type == "LinearImproperTorsions":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                periodicity = param[2].item()
                # Params 3 and 4 are phase1 and phase2
                idivf = param[5].item()
                k = k1 + k2
                if k == 0.0:
                    phase = 0.0
                else:
                    phase = math.acos((k1 - k2) / k)
                #                    phase = math.acos((k1 * math.cos(phase1) + k2 * math.cos(phase2))/k)
                dt = param.dtype
                new_params.append([k, periodicity, phase, idivf])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_torsion_units = (
                _KCAL_PER_MOL,
                _UNITLESS,
                _RADIANS,
                _UNITLESS,
            )
            reconstructed_torsion_cols = ("k", "periodicity", "phase", "idivf")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler("ImproperTorsions")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_torsion_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_torsion_cols)
                    }
                )
                _add_parameter_with_overwrite(handler, parameter_dict)

        elif potential.type == "vdW":
            handler = ff_smirnoff.get_parameter_handler(potential.type)

            # check if we have handler attributes to update
            attribute_names = potential.attribute_cols
            attribute_units = potential.attribute_units

            if potential.attributes is not None:
                opt_attributes = potential.attributes.detach().cpu().numpy()
                for j, (p, unit) in enumerate(zip(attribute_names, attribute_units)):
                    setattr(handler, p, opt_attributes[j] * unit)

            parameters_by_smarts: dict[str, dict[int | None, torch.Tensor]] = (
                collections.defaultdict(dict)
            )
            for parameter, parameter_key in zip(
                potential.parameters, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter

            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict: dict[str, str | Quantity] = {
                    "smirks": smarts,
                    "id": parameter_id,
                }
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * potential.parameter_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(potential.parameter_cols)
                    }
                )
                _add_parameter_with_overwrite(handler, parameter_dict)

            # ff_handler = ff_smirnoff.get_parameter_handler("vdW")

            # # check if we have handler attributes to update
            # attribute_names = potential.attribute_cols
            # attribute_units = potential.attribute_units

            # if potential.attributes is not None:
            #     opt_attributes = potential.attributes.detach().cpu().numpy()
            #     for j, (p, unit) in enumerate(zip(attribute_names, attribute_units)):
            #         setattr(ff_handler, p, opt_attributes[j] * unit)

            # parameter_names = potential.parameter_cols
            # parameter_units = potential.parameter_units

            # for i in range(len(potential.parameters)):
            #     smirks = potential.parameter_keys[i].id
            #     if "EP" in smirks:
            #         print(f"Skipping {smirks} as it is a virtual site")
            #         # skip fitted sites to dimers, we only have water and it should be 0 anyway
            #         continue
            #     ff_parameter = ff_handler[smirks]
            #     opt_parameters = potential.parameters[i].detach().cpu().numpy()
            #     for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
            #         setattr(ff_parameter, p, opt_parameters[j] * unit)

    return ff_smirnoff


def _create_smarts(mol: openff.toolkit.Molecule, idxs: torch.Tensor) -> str:
    """Create a mapped SMARTS representation of a molecule."""
    from rdkit import Chem

    mol_rdkit = mol.to_rdkit()

    for i, idx in enumerate(idxs):
        atom = mol_rdkit.GetAtomWithIdx(int(idx))
        atom.SetAtomMapNum(i + 1)

    smarts = Chem.MolToSmarts(mol_rdkit)
    return smarts


def _prepare_potential(
    mol: openff.toolkit.Molecule,
    symmetries: list[int],
    potential: smee.TensorPotential,
    parameter_map: smee.ValenceParameterMap,
    max_extend_distance: int = -1,
    excluded_smirks: list[str] | None = None,
) -> None:
    """Prepare a potential to use bespoke parameters for each 'slot'."""

    if not max_extend_distance == -1:
        raise NotImplementedError("max_extend_distance is not implemented yet.")

    excluded_smirks = excluded_smirks or []

    is_indexed = any(key.mult is not None for key in potential.parameter_keys)

    ids_to_parameter_idxs = collections.defaultdict(set)
    ids_to_particle_idxs = collections.defaultdict(set)

    ids_to_smarts = {}

    excluded_smirks_to_ids = {}

    if potential.type != "vdW":
        particle_idxs_list = parameter_map.particle_idxs
    else:
        particle_idxs_list = torch.tensor(
            [[i] for i in range(mol.n_atoms)], dtype=torch.long
        )
    for particle_idxs, assignment_row in zip(
        particle_idxs_list,
        parameter_map.assignment_matrix.to_dense(),
        strict=True,
    ):
        particle_idxs = tuple(int(idx) for idx in particle_idxs)
        particle_ids = tuple(symmetries[idx] for idx in particle_idxs)

        if potential.type != "ImproperTorsions" and particle_ids[-1] < particle_ids[0]:
            particle_ids = particle_ids[::-1]

        parameter_idxs = [
            parameter_idx
            for parameter_idx, value in enumerate(assignment_row)
            if int(value) != 0
        ]
        assert len(parameter_idxs) == 1

        initial_smarts = potential.parameter_keys[parameter_idxs[0]].id

        if initial_smarts in excluded_smirks:
            if initial_smarts not in excluded_smirks_to_ids:
                excluded_smirks_to_ids[initial_smarts] = particle_ids
            else:
                particle_ids = excluded_smirks_to_ids[initial_smarts]

        ids_to_parameter_idxs[particle_ids].add(parameter_idxs[0])
        ids_to_particle_idxs[particle_ids].add(particle_idxs)

        if potential.type == "ImproperTorsions":
            particle_idxs = (
                particle_idxs[1],
                particle_idxs[0],
                particle_idxs[2],
                particle_idxs[3],
            )

        if initial_smarts in excluded_smirks:
            ids_to_smarts[particle_ids] = initial_smarts
        else:
            ids_to_smarts[particle_ids] = _create_smarts(mol, particle_idxs)

    sorted_ids_to_parameter_idxs = {
        particle_ids: sorted(parameter_idxs)
        for particle_ids, parameter_idxs in ids_to_parameter_idxs.items()
    }

    parameter_ids = [
        (particle_ids, parameter_idx)
        for particle_ids, parameter_idxs in sorted_ids_to_parameter_idxs.items()
        for parameter_idx in parameter_idxs
    ]
    potential.parameters = potential.parameters[
        [parameter_idx for _, parameter_idx in parameter_ids]
    ]
    potential.parameter_keys = [
        openff.interchange.models.PotentialKey(
            id=ids_to_smarts[particle_ids],
            mult=(
                sorted_ids_to_parameter_idxs[particle_ids].index(parameter_idx)
                if is_indexed
                else None
            ),
            associated_handler=potential.type,
            bond_order=None,
            virtual_site_type=None,
            cosmetic_attributes={},
        )
        for particle_ids, parameter_idx in parameter_ids
    ]

    assignment_matrix = smee.utils.zeros_like(
        (len(particle_idxs_list), len(potential.parameters)),
        parameter_map.assignment_matrix,
    )

    particle_idxs_updated: list[tuple[int, ...]] = []

    for particle_ids, particle_idxs in ids_to_particle_idxs.items():
        for particle_idx in particle_idxs:
            for parameter_idx in sorted_ids_to_parameter_idxs[particle_ids]:
                j = parameter_ids.index((particle_ids, parameter_idx))

                assignment_matrix[len(particle_idxs_updated), j] = 1
                particle_idxs_updated.append(particle_idx)

    parameter_map.particle_idxs = smee.utils.tensor_like(
        particle_idxs_updated, particle_idxs_list
    )
    parameter_map.assignment_matrix = assignment_matrix.to_sparse()


def parameterise(
    settings: ParameterisationSettings,
    device: TorchDevice = "cuda",
) -> tuple[
    openff.toolkit.Molecule,
    openff.toolkit.ForceField,
    smee.TensorTopology,
    smee.TensorForceField,
]:
    """Prepare a Trainable object that contains a force field with
    unique parameters for each topologically symmetric term of a molecule.

    Parameters
    ----------
    settings: ParameterisationSettings
        The settings for the parameterisation.

    device: TorchDevice, default "cuda"
        The device to use for the force field and topology.

    Returns
    -------
    mol: openff.toolkit.Molecule
        The molecule that has been parameterised.
    off_ff: openff.toolkit.ForceField
        The original force field, used as a base for the bespoke force field.
    tensor_top: smee.TensorTopology
        The topology of the molecule.
    tensor_ff: smee.TensorForceField
        The force field with unique parameters for each topologically symmetric term.
    """
    mol = openff.toolkit.Molecule.from_smiles(
        settings.smiles, allow_undefined_stereo=True, hydrogens_are_explicit=False
    )
    off_ff = openff.toolkit.ForceField(settings.initial_force_field)

    if "[#1:1]-[*:2]" in off_ff["Constraints"].parameters:
        logger.warning(
            "The force field contains a constraint for [#1:1]-[*:2] which is not supported. "
            "Removing this constraint."
        )
        del off_ff["Constraints"].parameters["[#1:1]-[*:2]"]

    if settings.expand_torsions:
        torsion_generation_settings = settings.type_generation_settings.get(
            "ProperTorsions"
        )
        excluded_smirks = (
            torsion_generation_settings.exclude
            if torsion_generation_settings is not None
            else None
        )
        off_ff = _expand_torsions(off_ff, excluded_smirks=excluded_smirks)

    force_field, [topology] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(off_ff, mol.to_topology())
    )

    # Move the force field and topology to the requested device
    force_field = force_field.to(device)
    topology = topology.to(device)

    symmetries = list(Chem.CanonicalRankAtoms(mol.to_rdkit(), breakTies=False))
    if topology.n_v_sites != 0:
        raise NotImplementedError("virtual sites are not supported yet.")

    for (
        potential_type,
        type_generation_settings,
    ) in settings.type_generation_settings.items():
        potential = force_field.potentials_by_type.get(potential_type)

        if potential is None:
            logger.warning(
                f"No potential of type {potential_type} found in the force field. Skipping bespoke parameterisation for this type."
            )
            continue

        parameter_map = topology.parameters[potential_type]

        # Can only be a ValenceParameterMap here because we validate
        # that we only have valence terms in the settings
        parameter_map = cast(smee.ValenceParameterMap, parameter_map)

        _prepare_potential(
            mol,
            symmetries,
            potential,
            parameter_map,
            type_generation_settings.max_extend_distance,
            type_generation_settings.exclude,
        )

    if settings.linearise_harmonics:
        force_field = linearise_harmonics_force_field(force_field, device)
        topology = linearise_harmonics_topology(topology, device)

    return (
        mol,
        off_ff,
        topology,
        force_field,
    )


def _expand_torsions(
    ff: openff.toolkit.ForceField, excluded_smirks: list[str] | None = None
) -> openff.toolkit.ForceField:
    """Expand the torsion potential to include K0-4 for proper torsions"""
    excluded_smirks = excluded_smirks or []
    ff_copy = copy.deepcopy(ff)
    torsion_handler = ff_copy.get_parameter_handler("ProperTorsions")
    for parameter in torsion_handler:
        # Avoid expanding any excluded smarts
        if parameter.smirks in excluded_smirks:
            continue
        # set the defaults
        parameter.idivf = [1.0] * 4
        default_k = [0 * _KCAL_PER_MOL] * 4
        default_phase = [0 * _RADIANS] * 4
        default_p = [1, 2, 3, 4]
        # update the existing k values for the correct phase and p
        for i, p in enumerate(parameter.periodicity):
            try:
                default_k[p - 1] = parameter.k[i]
                default_phase[p - 1] = parameter.phase[i]
            except IndexError:
                continue
        # update with new parameters
        parameter.k = default_k
        parameter.phase = default_phase
        parameter.periodicity = default_p
    return ff_copy


def _add_angle_within_range(initial_angle: float, diff: float) -> float:
    """Add a difference to an angle cap to be within [0, pi]"""
    new_angle = initial_angle + diff
    if diff > 0:
        return min(new_angle, math.pi)
    else:
        return max(new_angle, 0.0)


def _compute_linear_harmonic_params(
    k: float,
    eq_value: float,
    compute_lower_bound: Callable[[float], float],
    compute_upper_bound: Callable[[float], float],
) -> tuple[float, float, float, float]:
    """Compute linearized harmonic parameters from standard parameters.

    This generic function distributes a force constant across two bounds,
    inversely proportional to the distance from each bound.

    Args:
        k: Force constant (e.g., kcal/mol/Å² or kcal/mol/rad²)
        eq_value: Equilibrium value (e.g., bond length or angle)
        compute_lower_bound: Function that takes eq_value and returns
            lower bound
        compute_upper_bound: Function that takes eq_value and returns
            upper bound

    Returns:
        Tuple of (k1, k2, eq1, eq2) where:
        - k1, k2: Distributed force constants
        - eq1, eq2: Lower and upper equilibrium value bounds
    """
    eq1 = compute_lower_bound(eq_value)
    eq2 = compute_upper_bound(eq_value)
    d = eq2 - eq1
    # Distribute force constant inversely proportional to distance from bounds
    k1 = k * (eq2 - eq_value) / d
    k2 = k * (eq_value - eq1) / d
    return k1, k2, eq1, eq2


def _linearize_bond_parameters(
    potential: smee.TensorPotential, device_type: str
) -> smee.TensorPotential:
    """Linearize bond potential parameters.

    Converts standard harmonic bond parameters (k, length) to linearized
    form (k1, k2, b1, b2) where the equilibrium bond length range is
    [0.5*length, 1.5*length].
    """
    new_potential = copy.deepcopy(potential)
    new_potential.type = "LinearBonds"
    new_potential.fn = "(k1+k2)/2*(r-(k1*length1+k2*length2)/(k1+k2))**2"
    new_potential.parameter_cols = ("k1", "k2", "b1", "b2")

    # Get dtype from the first parameter
    dtype = potential.parameters.dtype

    new_params = [
        _compute_linear_harmonic_params(
            param[0].item(),
            param[1].item(),
            lambda b: b - 0.4,  # Lower bound: current length - 0.4 Å
            lambda b: b + 0.4,  # Upper bound: current length + 0.4 Å
        )
        for param in potential.parameters
    ]

    new_potential.parameters = torch.tensor(
        new_params, dtype=dtype, requires_grad=False, device=device_type
    )
    new_potential.parameter_units = (
        _KCAL_PER_MOL_ANGSQ,
        _KCAL_PER_MOL_ANGSQ,
        _ANGSTROM,
        _ANGSTROM,
    )
    return new_potential


def _linearize_angle_parameters(
    potential: smee.TensorPotential, device_type: str
) -> smee.TensorPotential:
    """Linearize angle potential parameters.

    Converts standard harmonic angle parameters (k, angle) to linearized form
    (k1, k2, angle1, angle2) where the equilibrium angle range is [0, π].
    """
    new_potential = copy.deepcopy(potential)
    new_potential.type = "LinearAngles"
    new_potential.fn = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"
    new_potential.parameter_cols = ("k1", "k2", "angle1", "angle2")

    # Get dtype from the first parameter
    dtype = potential.parameters.dtype

    new_params = [
        _compute_linear_harmonic_params(
            param[0].item(),
            param[1].item(),
            lambda a: max(0.0, a - math.pi / 3),  # Lower bound: max(0, angle - π/3)
            lambda a: min(math.pi, a + math.pi / 3),  # Upper bound: min(π, angle + π/3)
        )
        for param in potential.parameters
    ]

    new_potential.parameters = torch.tensor(
        new_params, dtype=dtype, requires_grad=False, device=device_type
    )
    new_potential.parameter_units = (
        _KCAL_PER_MOL_RADSQ,
        _KCAL_PER_MOL_RADSQ,
        _RADIANS,
        _RADIANS,
    )
    return new_potential


def linearise_harmonics_force_field(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
    """Linearize the harmonic potential parameters in the forcefield.

    This converts Bonds and Angles potentials to their linearized forms
    (LinearBonds and LinearAngles) for more robust optimization.
    """
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []

    for potential in ff.potentials:
        if potential.type == "Bonds":
            ff_copy.potentials.append(
                _linearize_bond_parameters(potential, device_type)
            )
        elif potential.type == "Angles":
            ff_copy.potentials.append(
                _linearize_angle_parameters(potential, device_type)
            )
        else:
            ff_copy.potentials.append(potential)

    return ff_copy


def linearise_harmonics_topology(
    topology: smee.TensorTopology, device_type: TorchDevice
) -> smee.TensorTopology:
    """Linearize harmonic potential parameters in the topology.

    This updates the topology to use LinearBonds and LinearAngles
    parameter maps instead of Bonds and Angles.
    """
    topology_copy = topology.to(device_type)
    topology_copy.parameters["LinearBonds"] = copy.deepcopy(
        topology_copy.parameters["Bonds"]
    )
    topology_copy.parameters["LinearAngles"] = copy.deepcopy(
        topology.parameters["Angles"]
    )
    del topology_copy.parameters["Bonds"]
    del topology_copy.parameters["Angles"]
    return topology_copy
