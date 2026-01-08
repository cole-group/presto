"""
Unit tests for msm.py (for computing bond and angle parameters
using the modified Seminario method -- see https://doi.org/10.1021/acs.jctc.7b00785).
"""

import numpy as np
import pytest
from openff.toolkit import ForceField, Molecule

from bespokefit_smee.msm import (
    AngleParams,
    AngleScalingCalculator,
    BondParams,
    HessianDecomposer,
    _calculate_linear_angle_force_constant,
    _dot_product,
    _get_arbitrary_perpendicular,
    _mean_angle_params,
    _mean_bond_params,
    apply_msm_to_molecule,
    apply_msm_to_molecules,
    calculate_angle_force_constant,
    calculate_angle_params,
    calculate_bond_force_constant,
    calculate_bond_params,
    compute_perpendicular_vector_in_plane,
    unit_vector_along_bond,
    unit_vector_normal_to_plane,
)
from bespokefit_smee.settings import MSMSettings

# --- Fixtures ---


@pytest.fixture
def water_coords():
    """Coordinates for a water-like molecule with ~104 degree angle.

    Note: Coordinates are in nm to match the internal unit convention.
    """
    # O at origin, H atoms at ~104 degrees
    return np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.09572, 0.0, 0.0],  # H1 at 0.9572 Å (typical O-H)
            [-0.02399, 0.09270, 0.0],  # H2 to give ~104 degrees
        ]
    )


@pytest.fixture
def ethanol_coords():
    """Approximate coordinates for ethanol molecule.

    Note: Coordinates are in nm to match the internal unit convention.
    """
    # Rough ethanol geometry (9 atoms: C-C-O with 6 H)
    return np.array(
        [
            [0.0, 0.0, 0.0],  # C1
            [0.154, 0.0, 0.0],  # C2 (in nm)
            [0.216, 0.122, 0.0],  # O
            [-0.036, -0.051, -0.089],  # H1
            [-0.036, -0.051, 0.089],  # H2
            [-0.036, 0.102, 0.0],  # H3
            [0.19, -0.051, 0.089],  # H4
            [0.19, -0.051, -0.089],  # H5
            [0.31, 0.11, 0.0],  # H6 (on O)
        ]
    )


@pytest.fixture
def simple_3x3_hessian():
    """Simple 3x3 Hessian for a 1-atom system (for basic tests)."""
    return np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )


@pytest.fixture
def two_atom_hessian():
    """Simple 6x6 Hessian for a 2-atom diatomic system."""
    # Block structure for a simple spring along x-axis
    k = 500.0  # Force constant
    hessian = np.zeros((6, 6))
    # On-diagonal blocks
    hessian[0:3, 0:3] = np.diag([k, 0.0, 0.0])
    hessian[3:6, 3:6] = np.diag([k, 0.0, 0.0])
    # Off-diagonal blocks (coupling)
    hessian[0:3, 3:6] = np.diag([-k, 0.0, 0.0])
    hessian[3:6, 0:3] = np.diag([-k, 0.0, 0.0])
    return hessian


@pytest.fixture
def two_atom_coords():
    """Coordinates for a diatomic molecule along x-axis.

    Note: Coordinates are in nm to match the internal unit convention
    used by the MSM functions (Hessian is in kcal/mol/nm^2).
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],  # Bond length 0.1 nm (= 1.0 Å)
        ]
    )


@pytest.fixture
def msm_settings():
    """Default MSM settings for testing."""
    return MSMSettings()


# --- Unit Vector Tests ---


class TestUnitVectorAlongBond:
    """Tests for unit_vector_along_bond function."""

    def test_unit_vector_x_axis(self, two_atom_coords):
        """Test unit vector along x-axis."""
        u = unit_vector_along_bond(two_atom_coords, 0, 1)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(u, expected, atol=1e-10)

    def test_unit_vector_reversed(self, two_atom_coords):
        """Test unit vector is reversed when atom order is swapped."""
        u_forward = unit_vector_along_bond(two_atom_coords, 0, 1)
        u_backward = unit_vector_along_bond(two_atom_coords, 1, 0)
        np.testing.assert_allclose(u_forward, -u_backward, atol=1e-10)

    def test_unit_vector_is_normalized(self, water_coords):
        """Test that returned vector has unit length."""
        u = unit_vector_along_bond(water_coords, 0, 1)
        assert np.abs(np.linalg.norm(u) - 1.0) < 1e-10

    def test_diagonal_vector(self):
        """Test unit vector along a diagonal."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        u = unit_vector_along_bond(coords, 0, 1)
        expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        np.testing.assert_allclose(u, expected, atol=1e-10)


class TestUnitVectorNormalToPlane:
    """Tests for unit_vector_normal_to_plane function."""

    def test_perpendicular_to_xy_plane(self):
        """Test normal to xy plane is along z."""
        u_x = np.array([1.0, 0.0, 0.0])
        u_y = np.array([0.0, 1.0, 0.0])
        u_n = unit_vector_normal_to_plane(u_x, u_y)
        # Should be +/- z
        assert np.abs(np.abs(u_n[2]) - 1.0) < 1e-10
        assert np.abs(u_n[0]) < 1e-10
        assert np.abs(u_n[1]) < 1e-10

    def test_is_normalized(self):
        """Test that normal vector is normalized."""
        u1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        u2 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        u_n = unit_vector_normal_to_plane(u1, u2)
        assert np.abs(np.linalg.norm(u_n) - 1.0) < 1e-10

    def test_parallel_vectors_returns_perpendicular(self):
        """Test handling of parallel vectors."""
        u = np.array([1.0, 0.0, 0.0])
        u_n = unit_vector_normal_to_plane(u, u)
        # Should return arbitrary perpendicular, still normalized
        assert np.abs(np.linalg.norm(u_n) - 1.0) < 1e-10
        # Should be perpendicular to original
        assert np.abs(np.dot(u_n, u)) < 1e-10


class TestGetArbitraryPerpendicular:
    """Tests for _get_arbitrary_perpendicular function."""

    def test_perpendicular_to_x_axis(self):
        """Test perpendicular to x-axis."""
        u = np.array([1.0, 0.0, 0.0])
        perp = _get_arbitrary_perpendicular(u)
        assert np.abs(np.dot(perp, u)) < 1e-10
        assert np.abs(np.linalg.norm(perp) - 1.0) < 1e-10

    def test_perpendicular_to_y_axis(self):
        """Test perpendicular to y-axis."""
        u = np.array([0.0, 1.0, 0.0])
        perp = _get_arbitrary_perpendicular(u)
        assert np.abs(np.dot(perp, u)) < 1e-10
        assert np.abs(np.linalg.norm(perp) - 1.0) < 1e-10

    def test_perpendicular_to_diagonal(self):
        """Test perpendicular to diagonal vector."""
        u = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        perp = _get_arbitrary_perpendicular(u)
        assert np.abs(np.dot(perp, u)) < 1e-10
        assert np.abs(np.linalg.norm(perp) - 1.0) < 1e-10


class TestComputePerpendicularVectorInPlane:
    """Tests for compute_perpendicular_vector_in_plane function."""

    def test_water_angle(self, water_coords):
        """Test perpendicular vector for water H-O-H angle."""
        angle = (1, 0, 2)  # H1-O-H2
        u_pa = compute_perpendicular_vector_in_plane(water_coords, angle)
        # Should be perpendicular to the O-H1 bond
        u_oh1 = unit_vector_along_bond(water_coords, 0, 1)
        assert np.abs(np.dot(u_pa, u_oh1)) < 1e-10
        # Should be normalized
        assert np.abs(np.linalg.norm(u_pa) - 1.0) < 1e-10

    def test_linear_angle(self):
        """Test perpendicular vector for near-linear angle."""
        # Nearly linear A-B-C (180 degrees)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # A
                [1.0, 0.0, 0.0],  # B (central)
                [2.0, 0.001, 0.0],  # C (slightly off-axis)
            ]
        )
        angle = (0, 1, 2)
        u_pa = compute_perpendicular_vector_in_plane(coords, angle)
        # Should still be normalized
        assert np.abs(np.linalg.norm(u_pa) - 1.0) < 1e-10


# --- Dot Product Tests ---


class TestDotProduct:
    """Tests for _dot_product function."""

    def test_real_vectors(self):
        """Test dot product with real vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.5, 0.5, 0.0])
        result = _dot_product(u, v)
        assert np.abs(result - 0.5) < 1e-10

    def test_complex_eigenvector(self):
        """Test dot product with complex eigenvector."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([1.0 + 1.0j, 0.0, 0.0])
        result = _dot_product(u, v)
        # Conjugate: 1 - 1j, so result should be 1 - 1j
        expected = 1.0 - 1.0j
        assert np.abs(result - expected) < 1e-10

    def test_orthogonal_vectors(self):
        """Test dot product of orthogonal vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        result = _dot_product(u, v)
        assert np.abs(result) < 1e-10


# --- HessianDecomposer Tests ---


class TestHessianDecomposer:
    """Tests for HessianDecomposer class."""

    def test_decomposer_initialization(self, two_atom_hessian, two_atom_coords):
        """Test HessianDecomposer initializes correctly."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        assert decomposer.n_atoms == 2
        assert decomposer.eigenvals.shape == (2, 2, 3)
        assert decomposer.eigenvecs.shape == (3, 3, 2, 2)
        assert decomposer.bond_lengths.shape == (2, 2)

    def test_bond_lengths_computed(self, two_atom_hessian, two_atom_coords):
        """Test that bond lengths are computed correctly."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        # Bond length between atoms 0 and 1 should be 0.1 nm (coordinates are in nm)
        np.testing.assert_allclose(decomposer.bond_lengths[0, 1], 0.1, atol=1e-10)
        np.testing.assert_allclose(decomposer.bond_lengths[1, 0], 0.1, atol=1e-10)
        # Self distances should be 0
        assert np.abs(decomposer.bond_lengths[0, 0]) < 1e-10

    def test_eigenvalues_computed(self, two_atom_hessian, two_atom_coords):
        """Test that eigenvalues are computed."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        # The off-diagonal block [0,1] has [-k, 0, 0] on diagonal
        # Eigenvalues should be -k, 0, 0 (k=500)
        eigenvals = decomposer.eigenvals[0, 1, :]
        sorted_eigenvals = np.sort(np.real(eigenvals))
        assert np.abs(sorted_eigenvals[0] - (-500.0)) < 1e-10
        assert np.abs(sorted_eigenvals[1]) < 1e-10
        assert np.abs(sorted_eigenvals[2]) < 1e-10


# --- AngleScalingCalculator Tests ---


class TestAngleScalingCalculator:
    """Tests for AngleScalingCalculator class."""

    def test_single_angle(self, water_coords):
        """Test scaling factors for a single angle."""
        angles = [(1, 0, 2)]  # H1-O-H2
        calculator = AngleScalingCalculator(angles, water_coords, n_atoms=3)
        scalings = calculator.compute_scaling_factors()
        # Single angle with no neighbors should have scaling = (1.0, 1.0)
        assert len(scalings) == 1
        assert np.abs(scalings[0][0] - 1.0) < 1e-10
        assert np.abs(scalings[0][1] - 1.0) < 1e-10

    def test_multiple_angles_same_central(self, ethanol_coords):
        """Test scaling factors when multiple angles share a central atom."""
        # Create angles sharing C2 (index 1) as central
        angles = [
            (0, 1, 2),  # C1-C2-O
            (0, 1, 6),  # C1-C2-H4
            (0, 1, 7),  # C1-C2-H5
        ]
        calculator = AngleScalingCalculator(angles, ethanol_coords, n_atoms=9)
        scalings = calculator.compute_scaling_factors()
        # Should have scaling factors > 1 for angles sharing bonds
        assert len(scalings) == 3
        # All angles share C1-C2 bond, so first scaling should be > 1
        assert all(s[0] >= 1.0 for s in scalings)


# --- Bond Force Constant Tests ---


class TestCalculateBondForceConstant:
    """Tests for calculate_bond_force_constant function."""

    def test_simple_diatomic(self, two_atom_hessian, two_atom_coords):
        """Test force constant for simple diatomic."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond = (0, 1)
        k = calculate_bond_force_constant(
            bond,
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
        )
        # Force constant should be positive
        assert k > 0

    def test_symmetric_bond(self, two_atom_hessian, two_atom_coords):
        """Test that force constant is similar in both directions."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        k_01 = calculate_bond_force_constant(
            (0, 1),
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
        )
        k_10 = calculate_bond_force_constant(
            (1, 0),
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
        )
        # Should be similar (symmetric Hessian)
        np.testing.assert_allclose(k_01, k_10, rtol=0.1)


# --- Angle Force Constant Tests ---


class TestCalculateAngleForceConstant:
    """Tests for calculate_angle_force_constant function."""

    def test_returns_positive_force_constant(self, water_coords):
        """Test that force constant is positive."""
        # Create a mock Hessian with negative eigenvalues (as expected from bonded atoms)
        # For angle bending, we need negative eigenvalues in the off-diagonal blocks
        n_atoms = 3
        hessian = np.zeros((9, 9))
        # Fill with realistic spring-like values
        k_val = -500.0  # Negative for off-diagonal blocks
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [k_val, k_val, k_val]
                    )
                else:
                    # Positive on diagonal blocks
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [-k_val * 2, -k_val * 2, -k_val * 2]
                    )
        decomposer = HessianDecomposer(hessian, water_coords)

        angle = (1, 0, 2)  # H1-O-H2
        scalings = (1.0, 1.0)
        k_theta, theta_0 = calculate_angle_force_constant(
            angle,
            decomposer.bond_lengths,
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
            scalings,
        )
        assert k_theta > 0

    def test_equilibrium_angle_reasonable(self, water_coords):
        """Test that equilibrium angle is reasonable."""
        n_atoms = 3
        hessian = np.zeros((9, 9))
        k_val = -500.0
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [k_val, k_val, k_val]
                    )
                else:
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [-k_val * 2, -k_val * 2, -k_val * 2]
                    )
        decomposer = HessianDecomposer(hessian, water_coords)

        angle = (1, 0, 2)
        scalings = (1.0, 1.0)
        k_theta, theta_0 = calculate_angle_force_constant(
            angle,
            decomposer.bond_lengths,
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
            scalings,
        )
        # Water angle should be around 104 degrees, our mock coords
        # give approximately this
        assert 90.0 < theta_0 < 120.0


class TestCalculateLinearAngleForceConstant:
    """Tests for _calculate_linear_angle_force_constant function."""

    def test_linear_angle_returns_valid_result(self):
        """Test that linear angle calculation returns valid results."""
        # Linear configuration along x
        u_ab = np.array([1.0, 0.0, 0.0])
        u_cb = np.array([-1.0, 0.0, 0.0])  # Opposite direction (linear)
        bond_lens = (1.0, 1.0)
        # Simple eigenvalues/eigenvectors
        eigenvals = (
            np.array([100.0, 50.0, 50.0]),
            np.array([100.0, 50.0, 50.0]),
        )
        eigenvecs = (np.eye(3, dtype=complex), np.eye(3, dtype=complex))

        k_theta, theta_0 = _calculate_linear_angle_force_constant(
            u_ab, u_cb, bond_lens, eigenvals, eigenvecs, n_samples=50
        )
        # Force constant should be positive
        assert k_theta > 0
        # Angle should be close to 180 degrees
        assert np.abs(theta_0 - 180.0) < 1.0


# --- Parameter Calculation Tests ---


class TestCalculateBondParams:
    """Tests for calculate_bond_params function."""

    def test_returns_dict_of_bond_params(self, two_atom_hessian, two_atom_coords):
        """Test that function returns dictionary of BondParams."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond_indices = [(0, 1)]
        vib_scaling = 1.0

        result = calculate_bond_params(bond_indices, decomposer, vib_scaling)

        assert isinstance(result, dict)
        assert (0, 1) in result
        assert isinstance(result[(0, 1)], BondParams)

    def test_force_constant_scaled(self, two_atom_hessian, two_atom_coords):
        """Test that vibrational scaling is applied."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond_indices = [(0, 1)]

        result_1 = calculate_bond_params(bond_indices, decomposer, vib_scaling=1.0)
        result_2 = calculate_bond_params(bond_indices, decomposer, vib_scaling=0.5)

        # With scaling 0.5, force constant should be 0.25x
        ratio = result_2[(0, 1)].force_constant / result_1[(0, 1)].force_constant
        np.testing.assert_allclose(ratio, 0.25, rtol=0.01)

    def test_bond_length_preserved(self, two_atom_hessian, two_atom_coords):
        """Test that bond length is correctly reported in nm."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond_indices = [(0, 1)]

        result = calculate_bond_params(bond_indices, decomposer, vib_scaling=1.0)

        # Bond length is 1.0 Å = 0.1 nm
        np.testing.assert_allclose(result[(0, 1)].length, 0.1, rtol=0.01)


def _create_realistic_hessian(n_atoms: int, k_val: float = -500.0) -> np.ndarray:
    """Create a realistic mock Hessian for testing."""
    hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                    [k_val, k_val, k_val]
                )
            else:
                hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                    [-k_val * 2, -k_val * 2, -k_val * 2]
                )
    return hessian


class TestCalculateAngleParams:
    """Tests for calculate_angle_params function."""

    def test_returns_dict_of_angle_params(self, water_coords):
        """Test that function returns dictionary of AngleParams."""
        hessian = _create_realistic_hessian(3)
        decomposer = HessianDecomposer(hessian, water_coords)
        angle_indices = [(1, 0, 2)]
        vib_scaling = 1.0

        result = calculate_angle_params(angle_indices, decomposer, vib_scaling)

        assert isinstance(result, dict)
        assert (1, 0, 2) in result
        assert isinstance(result[(1, 0, 2)], AngleParams)

    def test_empty_angles_returns_empty_dict(self, water_coords):
        """Test that empty angle list returns empty dict."""
        hessian = _create_realistic_hessian(3)
        decomposer = HessianDecomposer(hessian, water_coords)

        result = calculate_angle_params([], decomposer, vib_scaling=1.0)

        assert result == {}

    def test_angle_in_radians(self, water_coords):
        """Test that angle is returned in radians."""
        hessian = _create_realistic_hessian(3)
        decomposer = HessianDecomposer(hessian, water_coords)
        angle_indices = [(1, 0, 2)]

        result = calculate_angle_params(angle_indices, decomposer, vib_scaling=1.0)

        # Angle should be in radians (water angle ~104 deg = ~1.8 rad)
        assert 1.5 < result[(1, 0, 2)].angle < 2.1


# --- Mean Parameter Tests ---


class TestMeanBondParams:
    """Tests for _mean_bond_params function."""

    def test_single_param(self):
        """Test mean of single parameter."""
        params = [BondParams(force_constant=1000.0, length=0.15)]
        result = _mean_bond_params(params)
        assert result.force_constant == 1000.0
        assert result.length == 0.15

    def test_multiple_params(self):
        """Test mean of multiple parameters."""
        params = [
            BondParams(force_constant=1000.0, length=0.14),
            BondParams(force_constant=2000.0, length=0.16),
        ]
        result = _mean_bond_params(params)
        np.testing.assert_allclose(result.force_constant, 1500.0, rtol=1e-10)
        np.testing.assert_allclose(result.length, 0.15, rtol=1e-10)

    def test_empty_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="No bond parameters"):
            _mean_bond_params([])


class TestMeanAngleParams:
    """Tests for _mean_angle_params function."""

    def test_single_param(self):
        """Test mean of single parameter."""
        params = [AngleParams(force_constant=500.0, angle=1.9)]
        result = _mean_angle_params(params)
        assert result.force_constant == 500.0
        assert result.angle == 1.9

    def test_multiple_params(self):
        """Test mean of multiple parameters."""
        params = [
            AngleParams(force_constant=400.0, angle=1.8),
            AngleParams(force_constant=600.0, angle=2.0),
        ]
        result = _mean_angle_params(params)
        assert result.force_constant == 500.0
        assert result.angle == 1.9

    def test_empty_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="No angle parameters"):
            _mean_angle_params([])


# --- Dataclass Tests ---


class TestBondParams:
    """Tests for BondParams dataclass."""

    def test_creation(self):
        """Test BondParams creation."""
        params = BondParams(force_constant=1000.0, length=0.15)
        assert params.force_constant == 1000.0
        assert params.length == 0.15

    def test_equality(self):
        """Test BondParams equality."""
        p1 = BondParams(force_constant=1000.0, length=0.15)
        p2 = BondParams(force_constant=1000.0, length=0.15)
        assert p1 == p2


class TestAngleParams:
    """Tests for AngleParams dataclass."""

    def test_creation(self):
        """Test AngleParams creation."""
        params = AngleParams(force_constant=500.0, angle=1.9)
        assert params.force_constant == 500.0
        assert params.angle == 1.9

    def test_equality(self):
        """Test AngleParams equality."""
        p1 = AngleParams(force_constant=500.0, angle=1.9)
        p2 = AngleParams(force_constant=500.0, angle=1.9)
        assert p1 == p2


# --- Integration Tests (marked as slow) ---


@pytest.mark.slow
class TestApplyMSMToMolecule:
    """Integration tests for apply_msm_to_molecule function."""

    def test_ethanol_molecule(self, msm_settings):
        """Test MSM on ethanol molecule."""
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        # Get bond and angle indices from force field
        labels = ff.label_molecules(mol.to_topology())[0]
        bond_indices = list(labels["Bonds"].keys())
        angle_indices = list(labels["Angles"].keys())

        bond_params, angle_params = apply_msm_to_molecule(
            mol, bond_indices, angle_indices, msm_settings
        )

        # Check that we got parameters for all bonds and angles
        assert len(bond_params) == len(bond_indices)
        assert len(angle_params) == len(angle_indices)

        # Check all force constants are positive
        for bp in bond_params.values():
            assert bp.force_constant > 0
            assert bp.length > 0

        for ap in angle_params.values():
            assert ap.force_constant > 0
            assert 0 < ap.angle < np.pi

    def test_returns_correct_types(self, msm_settings):
        """Test that return types are correct."""
        mol = Molecule.from_smiles("C")  # Methane - simple
        ff = ForceField("openff_unconstrained-2.2.1.offxml")
        labels = ff.label_molecules(mol.to_topology())[0]
        bond_indices = list(labels["Bonds"].keys())
        angle_indices = list(labels["Angles"].keys())

        bond_params, angle_params = apply_msm_to_molecule(
            mol, bond_indices, angle_indices, msm_settings
        )

        assert isinstance(bond_params, dict)
        assert isinstance(angle_params, dict)
        for k, v in bond_params.items():
            assert isinstance(k, tuple)
            assert isinstance(v, BondParams)


@pytest.mark.slow
class TestApplyMSMToMolecules:
    """Integration tests for apply_msm_to_molecules function."""

    def test_single_molecule(self, msm_settings):
        """Test MSM on single molecule."""
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        modified_ff = apply_msm_to_molecules([mol], ff, msm_settings)

        assert isinstance(modified_ff, ForceField)

    def test_multiple_molecules(self, msm_settings):
        """Test MSM on multiple molecules."""
        mols = [
            Molecule.from_smiles("CCO"),  # Ethanol
            Molecule.from_smiles("CC"),  # Ethane
        ]
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        modified_ff = apply_msm_to_molecules(mols, ff, msm_settings)

        assert isinstance(modified_ff, ForceField)

    def test_force_field_not_modified_in_place(self, msm_settings):
        """Test that original force field is not modified."""
        mol = Molecule.from_smiles("C")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        # Get original bond parameter
        bond_handler = ff.get_parameter_handler("Bonds")
        original_params = [(p.smirks, p.k, p.length) for p in bond_handler.parameters]

        modified_ff = apply_msm_to_molecules([mol], ff, msm_settings)

        # Check original is unchanged
        new_params = [(p.smirks, p.k, p.length) for p in bond_handler.parameters]
        assert original_params == new_params

    def test_modified_ff_has_different_params(self, msm_settings):
        """Test that modified force field has different parameters."""
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        # Get SMIRKS patterns used by the molecule
        labels = ff.label_molecules(mol.to_topology())[0]
        used_bond_smirks = {p.smirks for p in labels["Bonds"].values()}

        # Get original parameters
        bond_handler = ff.get_parameter_handler("Bonds")
        original_k = {
            p.smirks: p.k
            for p in bond_handler.parameters
            if p.smirks in used_bond_smirks
        }

        modified_ff = apply_msm_to_molecules([mol], ff, msm_settings)

        # Get modified parameters
        mod_bond_handler = modified_ff.get_parameter_handler("Bonds")
        modified_k = {
            p.smirks: p.k
            for p in mod_bond_handler.parameters
            if p.smirks in used_bond_smirks
        }

        # At least one parameter should be different
        # (MSM derives from MLP Hessian, different from original FF)
        different = any(original_k[s] != modified_k[s] for s in used_bond_smirks)
        assert different

    def test_units_preserved_from_original_ff(self, msm_settings):
        """Test that units from the original force field are preserved.

        Note: The `.to()` method may format units differently (e.g.,
        'kilocalorie_per_mole / angstrom ** 2' vs 'kilocalorie / angstrom ** 2 / mole')
        even though they are equivalent. So we check unit compatibility rather than
        exact string equality.
        """
        mol = Molecule.from_smiles("CCO")
        ff = ForceField("openff_unconstrained-2.2.1.offxml")

        # Get original units
        bond_handler = ff.get_parameter_handler("Bonds")
        angle_handler = ff.get_parameter_handler("Angles")

        # Store original units for one parameter of each type
        original_bond_k_units = bond_handler.parameters[0].k.units
        original_bond_length_units = bond_handler.parameters[0].length.units
        original_angle_k_units = angle_handler.parameters[0].k.units
        original_angle_angle_units = angle_handler.parameters[0].angle.units

        modified_ff = apply_msm_to_molecules([mol], ff, msm_settings)

        # Check units are compatible (dimensionally equivalent) in modified force field
        mod_bond_handler = modified_ff.get_parameter_handler("Bonds")
        mod_angle_handler = modified_ff.get_parameter_handler("Angles")

        for param in mod_bond_handler.parameters:
            # Check dimensional compatibility by attempting conversion
            assert param.k.is_compatible_with(
                original_bond_k_units
            ), f"Bond k units {param.k.units} not compatible with {original_bond_k_units}"
            assert param.length.is_compatible_with(
                original_bond_length_units
            ), f"Bond length units {param.length.units} not compatible with {original_bond_length_units}"

        for param in mod_angle_handler.parameters:
            assert param.k.is_compatible_with(
                original_angle_k_units
            ), f"Angle k units {param.k.units} not compatible with {original_angle_k_units}"
            assert param.angle.is_compatible_with(
                original_angle_angle_units
            ), f"Angle units {param.angle.units} not compatible with {original_angle_angle_units}"


# --- MSMSettings Tests ---


class TestMSMSettings:
    """Tests for MSMSettings class."""

    def test_default_settings(self):
        """Test default MSM settings."""
        settings = MSMSettings()
        assert settings.vib_scaling == 0.957
        assert settings.ml_potential == "egret-1"

    def test_custom_vib_scaling(self):
        """Test custom vibrational scaling."""
        settings = MSMSettings(vib_scaling=1.0)
        assert settings.vib_scaling == 1.0

    def test_custom_ml_potential(self):
        """Test custom ML potential."""
        settings = MSMSettings(ml_potential="mace-off23-medium")
        assert settings.ml_potential == "mace-off23-medium"
