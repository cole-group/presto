#!/usr/bin/env python
"""
Generate reference MSM values using QUBEKit's ModSeminario implementation.

This script uses QUBEKit directly to compute bond and angle parameters using
the Modified Seminario Method, which can then be used as reference values
for testing our implementation.

Usage:
    conda run -n qubekit python bespokefit_smee/tests/unit/generate_qubekit_reference.py

Requirements:
    - QUBEKit must be installed (available in the 'qubekit' conda environment)

The script will:
    1. Create an asymmetric halogenated molecule with QUBEKit
    2. Generate a mock Hessian matrix
    3. Run QUBEKit's ModSeminario method
    4. Output the resulting bond and angle parameters

These values can be compared against our implementation to verify correctness.

Note: We use an asymmetric molecule (2-bromo-2-chloroethanol, BrClCH-CH2-OH) to avoid
symmetry averaging issues when comparing against our implementation.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Check if QUBEKit is available
try:
    from qubekit.bonded import ModSeminario
    from qubekit.molecules import Ligand
    from qubekit.utils import constants
except ImportError:
    print("ERROR: QUBEKit is not installed in this environment.")
    print("Please run this script using:")
    print("    conda run -n qubekit python bespokefit_smee/tests/unit/generate_qubekit_reference.py")
    sys.exit(1)


def create_mock_hessian_angstrom(n_atoms: int, k_diagonal: float = 500.0) -> np.ndarray:
    """Create a mock Hessian matrix in kcal/mol/Å² units.

    This creates a simple diagonal-dominated Hessian that represents
    harmonic restoring forces.

    Args:
        n_atoms: Number of atoms
        k_diagonal: Force constant for diagonal elements (kcal/mol/Å²)

    Returns:
        Hessian matrix of shape (3*n_atoms, 3*n_atoms) in kcal/mol/Å²
    """
    size = 3 * n_atoms
    hessian = np.zeros((size, size))

    # Set diagonal blocks (self-interaction)
    for i in range(n_atoms):
        block = np.diag([k_diagonal, k_diagonal, k_diagonal])
        hessian[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = block

    # Set off-diagonal blocks (interactions between atoms)
    # Use a smaller coupling constant
    k_coupling = -k_diagonal / (n_atoms - 1)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                # Simple coupling along the bond direction would be more realistic,
                # but for testing we use a simple isotropic coupling
                hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                    [k_coupling, k_coupling, k_coupling]
                )

    # Ensure symmetry
    hessian = 0.5 * (hessian + hessian.T)

    return hessian


def create_mock_hessian_atomic_units(n_atoms: int, k_diagonal: float = 500.0) -> np.ndarray:
    """Create a mock Hessian matrix in atomic units (Hartree/Bohr²).
    
    QUBEKit expects the Hessian in atomic units and converts internally.
    
    Args:
        n_atoms: Number of atoms
        k_diagonal: Force constant for diagonal elements in kcal/mol/Å²
        
    Returns:
        Hessian matrix in Hartree/Bohr²
    """
    # First create in kcal/mol/Å²
    hessian_kcal_angstrom = create_mock_hessian_angstrom(n_atoms, k_diagonal)
    
    # Convert to atomic units (Hartree/Bohr²)
    # QUBEKit does: hessian *= constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS**2)
    # So to go backwards: hessian_au = hessian_kcal_A2 / (HA_TO_KCAL_P_MOL / BOHR_TO_ANGS**2)
    conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS**2)
    hessian_au = hessian_kcal_angstrom / conversion
    
    return hessian_au


def main():
    """Generate reference values using QUBEKit's ModSeminario."""
    print("=" * 70)
    print("QUBEKit Modified Seminario Method - Reference Value Generator")
    print("=" * 70)
    print()
    
    # Create a fully asymmetric halogenated molecule using QUBEKit
    # Fluorochlorobromomethanol: FC(Cl)(Br)O - all atoms unique, no symmetry
    # SMILES: OC(F)(Cl)Br
    print("Creating fluorochlorobromomethanol molecule (fully asymmetric)...")
    mol = Ligand.from_smiles("OC(F)(Cl)Br", "fluorochlorobromomethanol")
    
    print(f"  Number of atoms: {mol.n_atoms}")
    print(f"  Number of bonds: {mol.n_bonds}")
    print(f"  Number of angles: {mol.n_angles}")
    print()
    
    # Print coordinates
    print("Coordinates (Angstroms):")
    for i, (atom, coord) in enumerate(zip(mol.atoms, mol.coordinates)):
        print(f"  {i}: {atom.atomic_symbol:2s} [{coord[0]:10.6f}, {coord[1]:10.6f}, {coord[2]:10.6f}]")
    print()
    
    # Print bonds
    print("Bonds:")
    for bond in mol.bonds:
        print(f"  ({bond.atom1_index}, {bond.atom2_index})")
    print()
    
    # Print angles
    print("Angles:")
    for angle in mol.angles:
        print(f"  {angle}")
    print()
    
    # Create mock Hessian in atomic units (QUBEKit's expected input)
    print("Creating mock Hessian in atomic units (Hartree/Bohr²)...")
    hessian_au = create_mock_hessian_atomic_units(mol.n_atoms, k_diagonal=500.0)
    mol.hessian = hessian_au
    print(f"  Hessian shape: {hessian_au.shape}")
    print()
    
    # Run ModSeminario
    print("Running QUBEKit ModSeminario...")
    mod_sem = ModSeminario(vibrational_scaling=1.0)
    mol = mod_sem.run(molecule=mol)
    print("  Done!")
    print()
    
    # Extract and print bond parameters
    print("=" * 70)
    print("BOND PARAMETERS (QUBEKit output)")
    print("=" * 70)
    print("Units: length in nm, k in kJ/mol/nm²")
    print()
    
    bond_results = {}
    for bond in mol.bonds:
        bond_key = (bond.atom1_index, bond.atom2_index)
        param = mol.BondForce[bond_key]
        bond_results[str(bond_key)] = {
            "length_nm": param.length,
            "k_kj_mol_nm2": param.k,
        }
        print(f"  Bond {bond_key}:")
        print(f"    length = {param.length:.6f} nm")
        print(f"    k = {param.k:.2f} kJ/mol/nm²")
    print()
    
    # Extract and print angle parameters
    print("=" * 70)
    print("ANGLE PARAMETERS (QUBEKit output)")
    print("=" * 70)
    print("Units: angle in radians, k in kJ/mol/rad²")
    print()
    
    angle_results = {}
    for angle in mol.angles:
        param = mol.AngleForce[angle]
        angle_results[str(angle)] = {
            "angle_rad": param.angle,
            "angle_deg": np.degrees(param.angle),
            "k_kj_mol_rad2": param.k,
        }
        print(f"  Angle {angle}:")
        print(f"    angle = {param.angle:.6f} rad ({np.degrees(param.angle):.2f}°)")
        print(f"    k = {param.k:.2f} kJ/mol/rad²")
    print()
    
    # Print summary as Python dict for copy-paste
    print("=" * 70)
    print("PYTHON REFERENCE DATA (copy-paste into test file)")
    print("=" * 70)
    print()
    
    # Coordinates
    print("# Ethanol coordinates in Angstroms (QUBEKit native format)")
    print("ETHANOL_COORDS_ANGSTROM = np.array([")
    for coord in mol.coordinates:
        print(f"    [{coord[0]:12.8f}, {coord[1]:12.8f}, {coord[2]:12.8f}],")
    print("])")
    print()
    
    # Bonds
    print("# Ethanol bonds (0-indexed atom pairs)")
    print("ETHANOL_BONDS = [")
    for bond in mol.bonds:
        print(f"    ({bond.atom1_index}, {bond.atom2_index}),")
    print("]")
    print()
    
    # Angles
    print("# Ethanol angles (central atom is middle index)")
    print("ETHANOL_ANGLES = [")
    for angle in mol.angles:
        print(f"    {angle},")
    print("]")
    print()
    
    # Bond reference values
    print("# QUBEKit bond parameters")
    print("# Units: length in nm, k in kJ/mol/nm² (OpenMM convention: U = k*(r-r0)²)")
    print("QUBEKIT_BOND_PARAMS = {")
    for bond in mol.bonds:
        bond_key = (bond.atom1_index, bond.atom2_index)
        param = mol.BondForce[bond_key]
        print(f"    {bond_key}: {{'length': {param.length:.10f}, 'k': {param.k:.6f}}},")
    print("}")
    print()
    
    # Angle reference values
    print("# QUBEKit angle parameters")
    print("# Units: angle in degrees, k in kJ/mol/rad² (OpenMM convention: U = k*(theta-theta0)²)")
    print("QUBEKIT_ANGLE_PARAMS = {")
    for angle in mol.angles:
        param = mol.AngleForce[angle]
        print(f"    {angle}: {{'angle': {np.degrees(param.angle):.10f}, 'k': {param.k:.6f}}},")
    print("}")
    print()
    
    # Save to JSON file
    output_data = {
        "coordinates_angstrom": mol.coordinates.tolist(),
        "bonds": [(b.atom1_index, b.atom2_index) for b in mol.bonds],
        "angles": list(mol.angles),
        "bond_params": bond_results,
        "angle_params": angle_results,
        "notes": {
            "hessian_type": "mock_diagonal_dominated",
            "hessian_k_diagonal_kcal_mol_A2": 500.0,
            "vibrational_scaling": 1.0,
            "units": {
                "length": "nm",
                "bond_k": "kJ/mol/nm² (OpenMM convention: U = k*(r-r0)²)",
                "angle": "radians (also provided in degrees)",
                "angle_k": "kJ/mol/rad² (OpenMM convention: U = k*(theta-theta0)²)",
            },
        },
    }
    
    output_path = Path(__file__).parent / "qubekit_reference_values.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Reference values saved to: {output_path}")
    print()
    
    print("=" * 70)
    print("UNIT CONVERSION NOTES")
    print("=" * 70)
    print("""
QUBEKit internal workflow:
1. Input Hessian is in atomic units (Hartree/Bohr²)
2. Converts to kcal/mol/Å² using: hessian *= HA_TO_KCAL_P_MOL / BOHR_TO_ANGS²
3. ModSeminario calculates force constants in kcal/mol/Å² (bonds) or kcal/mol/rad² (angles)
4. Output is converted to OpenMM units:
   - Bonds: kJ/mol/nm² using KCAL_TO_KJ * 200 (= 4.184 * 200 = 836.8)
     Factor of 200 = 100 (Å² → nm²) × 2 (potential convention)
   - Angles: kJ/mol/rad² using KCAL_TO_KJ * 2 (= 4.184 * 2 = 8.368)
     Factor of 2 is for potential convention

OpenMM convention: U = k*(r-r0)² (no 1/2 factor)
OpenFF/SMIRNOFF convention: U = (k/2)*(r-r0)² (has 1/2 factor)

So QUBEKit k values are 2× larger than OpenFF k values for the same physical potential.
""")


if __name__ == "__main__":
    main()
