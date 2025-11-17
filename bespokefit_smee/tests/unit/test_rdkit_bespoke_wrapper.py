"""Comprehensive tests for RDKit Bespoke Toolkit Wrapper."""

import threading
from pathlib import Path

import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.utils.exceptions import ChargeMethodUnavailableError
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

from bespokefit_smee.utils.rdkit_bespoke_wrapper import (
    RDKitBespokeToolkitWrapper,
    use_bespoke_rdkit_toolkit,
    use_bespoke_rdkit_toolkit_decorator,
)


class TestRDKitBespokeToolkitWrapper:
    """Test the custom toolkit wrapper class."""

    def test_toolkit_name(self):
        """Test that the toolkit has a distinct name."""
        wrapper = RDKitBespokeToolkitWrapper()
        assert "Bespoke" in wrapper._toolkit_name
        assert "RDKit" in wrapper._toolkit_name

    def test_charge_methods_disabled(self):
        """Test that charge assignment methods are disabled."""
        wrapper = RDKitBespokeToolkitWrapper()
        assert wrapper._supported_charge_methods == {}

    def test_charge_assignment_raises_error(self):
        """Test that charge assignment raises an informative error."""
        wrapper = RDKitBespokeToolkitWrapper()
        molecule = Molecule.from_smiles("CCO")

        with pytest.raises(ChargeMethodUnavailableError) as exc_info:
            wrapper.assign_partial_charges(molecule, partial_charge_method="am1bcc")

        error_msg = str(exc_info.value)
        assert "RDKitBespokeToolkitWrapper does not support" in error_msg
        assert "AmberToolsToolkitWrapper" in error_msg
        assert "SMARTS matching" in error_msg

    def test_repr(self):
        """Test the string representation."""
        wrapper = RDKitBespokeToolkitWrapper()
        repr_str = repr(wrapper)
        assert "RDKitBespokeToolkitWrapper" in repr_str
        assert "symmetry" in repr_str.lower()


class TestContextManager:
    """Test the use_bespoke_rdkit_toolkit context manager."""

    def test_context_manager_registers_bespoke_toolkit(self):
        """Test that the context manager adds the bespoke wrapper."""
        original_toolkits = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)

        with use_bespoke_rdkit_toolkit():
            # Check that bespoke toolkit is present
            has_bespoke = any(
                isinstance(t, RDKitBespokeToolkitWrapper)
                for t in GLOBAL_TOOLKIT_REGISTRY._toolkits
            )
            assert has_bespoke, "Bespoke toolkit should be registered in context"

        # After exiting, registry should be restored
        assert GLOBAL_TOOLKIT_REGISTRY._toolkits == original_toolkits

    def test_context_manager_moves_bespoke_to_front(self):
        """Test that the bespoke wrapper is placed at the front."""
        with use_bespoke_rdkit_toolkit():
            first_toolkit = GLOBAL_TOOLKIT_REGISTRY._toolkits[0]
            assert isinstance(first_toolkit, RDKitBespokeToolkitWrapper), (
                "Bespoke toolkit should be first in the registry"
            )

    def test_context_manager_restores_on_exception(self):
        """Test that the registry is restored even if an exception occurs."""
        original_toolkits = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)

        with pytest.raises(ValueError):
            with use_bespoke_rdkit_toolkit():
                raise ValueError("Test exception")

        # Registry should still be restored
        assert GLOBAL_TOOLKIT_REGISTRY._toolkits == original_toolkits

    def test_context_manager_idempotent(self):
        """Test that using the context manager twice doesn't add duplicates."""
        with use_bespoke_rdkit_toolkit():
            first_count = sum(
                isinstance(t, RDKitBespokeToolkitWrapper)
                for t in GLOBAL_TOOLKIT_REGISTRY._toolkits
            )

            # Nested context
            with use_bespoke_rdkit_toolkit():
                second_count = sum(
                    isinstance(t, RDKitBespokeToolkitWrapper)
                    for t in GLOBAL_TOOLKIT_REGISTRY._toolkits
                )

                # Should still only have one bespoke instance
                assert first_count == second_count == 1

    def test_context_manager_yields_registry(self):
        """Test that the context manager yields the registry."""
        with use_bespoke_rdkit_toolkit() as registry:
            assert registry is GLOBAL_TOOLKIT_REGISTRY

    def test_context_manager_thread_safe(self):
        """Test that the context manager is thread-safe."""
        errors = []

        def use_context():
            try:
                original = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)
                with use_bespoke_rdkit_toolkit():
                    pass
                assert GLOBAL_TOOLKIT_REGISTRY._toolkits == original
            except AssertionError as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = [threading.Thread(target=use_context) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety issues: {errors}"


class TestDecorator:
    """Test the use_bespoke_rdkit_toolkit_decorator."""

    def test_decorator_without_parentheses(self):
        """Test decorator can be used without parentheses."""
        original_toolkits = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)

        @use_bespoke_rdkit_toolkit_decorator
        def dummy_func():
            first = GLOBAL_TOOLKIT_REGISTRY._toolkits[0]
            return isinstance(first, RDKitBespokeToolkitWrapper)

        result = dummy_func()
        assert result, "Bespoke toolkit should be active in decorated function"
        assert GLOBAL_TOOLKIT_REGISTRY._toolkits == original_toolkits, (
            "Registry should be restored after function"
        )

    def test_decorator_with_parentheses(self):
        """Test decorator can be used with parentheses."""
        original_toolkits = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)

        # Note: currently decorator doesn't support being called with ()
        # This test documents expected behavior if we add that feature
        @use_bespoke_rdkit_toolkit_decorator
        def dummy_func():
            first = GLOBAL_TOOLKIT_REGISTRY._toolkits[0]
            return isinstance(first, RDKitBespokeToolkitWrapper)

        result = dummy_func()
        assert result, "Bespoke toolkit should be active in decorated function"
        assert GLOBAL_TOOLKIT_REGISTRY._toolkits == original_toolkits, (
            "Registry should be restored after function"
        )

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @use_bespoke_rdkit_toolkit_decorator
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == """My docstring."""

    def test_decorator_with_arguments(self):
        """Test decorator works with functions that take arguments."""

        @use_bespoke_rdkit_toolkit_decorator
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_with_return_value(self):
        """Test decorator preserves return values."""

        @use_bespoke_rdkit_toolkit_decorator
        def get_value():
            return 42

        assert get_value() == 42

    def test_decorator_rejects_async_functions(self):
        """Test that decorator raises TypeError for async functions."""
        with pytest.raises(TypeError, match="does not support async"):

            @use_bespoke_rdkit_toolkit_decorator
            async def async_func():
                pass


class TestSMARTSMatchingSimpleMols:
    """Test SMARTS matching with simple molecules."""

    @pytest.fixture
    def benzene(self):
        """Benzene molecule - highly symmetric."""
        return Molecule.from_smiles("c1ccccc1"), "[c:1]1[c:2][c:3][c:4][c:5][c:6]1"

    @pytest.fixture
    def methane(self):
        """Methane molecule - tetrahedral symmetry."""
        return Molecule.from_smiles("C"), "[C:1]([H:2])([H:3])([H:4])[H:5]"

    @pytest.fixture
    def ethanol(self):
        """Ethanol molecule - lower symmetry."""
        return Molecule.from_smiles("CCO"), "[C:1][C:2][O:3]"

    @pytest.mark.parametrize("molecule", ["benzene", "methane", "ethanol"])
    def test_symmetry_matching(self, request, molecule):
        """Test that we get the same number of matches for a few simple molecules."""
        molecule, smarts = request.getfixturevalue(molecule)

        bespoke = RDKitBespokeToolkitWrapper()
        default = RDKitToolkitWrapper()

        bespoke_matches = bespoke.find_smarts_matches(molecule, smarts)
        default_matches = default.find_smarts_matches(molecule, smarts)

        assert len(bespoke_matches) > 0
        assert bespoke_matches == default_matches


class TestParameterAssignmentEquivalence:
    """
    Test parameter assignment is identical between wrappers.

    This is the critical test: ensure the bespoke wrapper produces
    the same force field parameters as the default wrapper.
    """

    @pytest.fixture
    def jnk1_molecule(self):
        """JNK1 ligand molecule - relatively complex, some symmetry, linear torsion."""
        smiles = (
            "C(C(C([H])([H])[H])(Oc1nc(c(c(N([H])[H])c1C#N)[H])"
            "N(C(=O)C(c1c(OC([H])([H])[H])c(c(S(=O)(=O)C([H])([H])[H])"
            "c(c1[H])OC([H])([H])[H])[H])([H])[H])[H])[H])([H])([H])[H]"
        )
        return Molecule.from_smiles(smiles)

    @pytest.fixture
    def ethanol_molecule(self):
        return Molecule.from_smiles("CCO")

    @pytest.fixture
    def benzene_molecule(self):
        return Molecule.from_smiles("c1ccccc1")

    @pytest.fixture
    def acetic_acid_molecule(self):
        return Molecule.from_smiles("CC(=O)O")

    @pytest.fixture
    def isobutane_molecule(self):
        return Molecule.from_smiles("CC(C)C")

    @pytest.fixture
    def cyclohexane_molecule(self):
        return Molecule.from_smiles("C1CCCCC1")

    @pytest.fixture
    def openff_ff(self):
        """OpenFF unconstrained force field."""
        return ForceField("openff_unconstrained-2.3.0-rc2.offxml")

    @pytest.fixture
    def jnk1_bespoke_ff(self):
        """JNK1 bespoke force field."""
        ff_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "force_fields"
            / "jnk1_bespoke_ff.offxml"
        )
        return ForceField(str(ff_path))

    def _get_parameter_assignments(self, molecule, force_field, toolkit):
        """
        Get all parameter assignments for a molecule using a toolkit.

        Returns dict mapping parameter type to list of assignments.
        """
        # Temporarily modify the registry
        original_toolkits = list(GLOBAL_TOOLKIT_REGISTRY._toolkits)
        try:
            # Clear and add only the specified toolkit
            GLOBAL_TOOLKIT_REGISTRY._toolkits.clear()
            GLOBAL_TOOLKIT_REGISTRY._toolkits.append(toolkit)

            # Create parameter assignments
            topology = molecule.to_topology()
            labels = force_field.label_molecules(topology)[0]

            return labels

        finally:
            # Restore original registry
            GLOBAL_TOOLKIT_REGISTRY._toolkits[:] = original_toolkits

    @pytest.mark.parametrize(
        "mol,force_field",
        [
            ("ethanol_molecule", "openff_ff"),
            ("benzene_molecule", "openff_ff"),
            ("acetic_acid_molecule", "openff_ff"),
            ("isobutane_molecule", "openff_ff"),
            ("cyclohexane_molecule", "openff_ff"),
            ("jnk1_molecule", "openff_ff"),
            # ("jnk1_molecule", "jnk1_bespoke_ff"), # Passes, but super slow
        ],
    )
    def test_parameters_identical(self, request, mol, force_field):
        """Test that all assigned parameters are identical between toolkits."""

        mol = request.getfixturevalue(mol)
        force_field = request.getfixturevalue(force_field)

        bespoke_toolkit = RDKitBespokeToolkitWrapper()
        default_toolkit = RDKitToolkitWrapper()

        bespoke_labels = self._get_parameter_assignments(
            mol, force_field, bespoke_toolkit
        )
        default_labels = self._get_parameter_assignments(
            mol, force_field, default_toolkit
        )

        # Should have the same parameter types
        assert set(bespoke_labels.keys()) == set(default_labels.keys())

        # Compare all parameter types
        for param_type in bespoke_labels.keys():
            bespoke_params = bespoke_labels[param_type]
            default_params = default_labels[param_type]

            # Same number of parameters assigned
            assert len(bespoke_params) == len(default_params), (
                f"Parameter type {param_type}: "
                f"bespoke has {len(bespoke_params)}, "
                f"default has {len(default_params)}"
            )

            # Each parameter assignment should match
            for idx in bespoke_params.keys():
                assert idx in default_params, (
                    f"Parameter type {param_type}: "
                    f"index {idx} in bespoke but not default"
                )
                assert bespoke_params[idx].id == default_params[idx].id, (
                    f"Parameter type {param_type} at {idx}: "
                    f"bespoke={bespoke_params[idx].id}, "
                    f"default={default_params[idx].id}"
                )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_smarts_raises_error(self):
        """Test that invalid SMARTS raises appropriate error."""
        from openff.toolkit.utils.exceptions import (
            ChemicalEnvironmentParsingError,
        )

        bespoke = RDKitBespokeToolkitWrapper()
        molecule = Molecule.from_smiles("CCO")

        with pytest.raises(ChemicalEnvironmentParsingError):
            bespoke.find_smarts_matches(molecule, "[invalid smarts")

    def test_empty_molecule(self):
        """Test behavior with an empty molecule."""
        # OpenFF Toolkit may not support truly empty molecules,
        # but we can test minimal molecules
        molecule = Molecule.from_smiles("[H][H]")  # H2
        bespoke = RDKitBespokeToolkitWrapper()

        # Should not crash
        matches = bespoke.find_smarts_matches(molecule, "[H:1][H:2]")
        assert len(matches) >= 1


# class TestPerformance:
#     """Test performance characteristics (informational/benchmark tests)."""

#     @pytest.mark.slow
#     def test_highly_symmetric_molecule_performance(self):
#         """
#         Benchmark showing bespoke wrapper is faster for symmetric
#         molecules.

#         This test is marked 'slow' and primarily serves as
#         documentation.
#         """
#         import time

#         # Create a highly symmetric molecule (adamantane-like or cubane)
#         # Using a simpler symmetric molecule for testing
#         # Bicyclic, symmetric
#         molecule = Molecule.from_smiles(
#             "C1(C2CCC1C2)C3CCC3", allow_undefined_stereo=True
#         )

#         # SMARTS matching all atoms
#         smarts = "[C:1]1([C:2]2[C:3][C:4][C:5]1[C:6]2)[C:7]3[C:8][C:9][C:10]3"

#         bespoke = RDKitBespokeToolkitWrapper()
#         default = RDKitToolkitWrapper()

#         # Time bespoke
#         start = time.time()
#         bespoke_matches = bespoke.find_smarts_matches(molecule, smarts)
#         bespoke_time = time.time() - start

#         # Time default
#         start = time.time()
#         default_matches = default.find_smarts_matches(molecule, smarts)
#         default_time = time.time() - start

#         print(f"\nBespoke: {len(bespoke_matches)} matches in {bespoke_time:.4f}s")
#         print(f"Default: {len(default_matches)} matches in {default_time:.4f}s")
#         breakpoint()

#         # Informational: bespoke should be faster
#         assert len(bespoke_matches) == len(default_matches)
