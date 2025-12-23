"""Unit tests for data_utils module."""

import numpy as np
import pytest
import torch

from bespokefit_smee.data_utils import (
    WEIGHTED_DATA_SCHEMA,
    WeightedEntry,
    create_dataset_with_uniform_weights,
    create_weighted_dataset,
    get_weights_from_entry,
    has_weights,
    merge_weighted_datasets,
)


class TestWeightedDataSchema:
    """Tests for the WEIGHTED_DATA_SCHEMA."""

    def test_schema_has_required_fields(self):
        """Test that schema has all required fields."""
        field_names = [field.name for field in WEIGHTED_DATA_SCHEMA]
        assert "smiles" in field_names
        assert "coords" in field_names
        assert "energy" in field_names
        assert "forces" in field_names
        assert "energy_weights" in field_names
        assert "forces_weights" in field_names

    def test_schema_field_count(self):
        """Test that schema has exactly 6 fields."""
        assert len(WEIGHTED_DATA_SCHEMA) == 6


class TestCreateWeightedDataset:
    """Tests for create_weighted_dataset function."""

    @pytest.fixture
    def simple_entry(self) -> WeightedEntry:
        """Create a simple weighted entry for testing."""
        n_confs = 3
        n_atoms = 5
        return WeightedEntry(
            smiles="CCO",
            coords=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
            energy=torch.rand(n_confs, dtype=torch.float64),
            forces=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
            energy_weights=torch.ones(n_confs, dtype=torch.float64),
            forces_weights=torch.ones(n_confs, dtype=torch.float64),
        )

    def test_creates_dataset_from_single_entry(self, simple_entry):
        """Test that a dataset is created from a single entry."""
        dataset = create_weighted_dataset([simple_entry])
        assert len(dataset) == 1

    def test_dataset_has_correct_smiles(self, simple_entry):
        """Test that dataset preserves SMILES."""
        dataset = create_weighted_dataset([simple_entry])
        assert dataset[0]["smiles"] == "CCO"

    def test_dataset_has_torch_format(self, simple_entry):
        """Test that dataset uses torch format."""
        dataset = create_weighted_dataset([simple_entry])
        entry = dataset[0]
        assert isinstance(entry["coords"], torch.Tensor)
        assert isinstance(entry["energy"], torch.Tensor)
        assert isinstance(entry["forces"], torch.Tensor)
        assert isinstance(entry["energy_weights"], torch.Tensor)
        assert isinstance(entry["forces_weights"], torch.Tensor)

    def test_coords_flattened_and_restored(self, simple_entry):
        """Test that coordinates are properly flattened and can be restored."""
        n_confs = 3
        n_atoms = 5
        dataset = create_weighted_dataset([simple_entry])
        entry = dataset[0]

        coords_restored = entry["coords"].reshape(n_confs, n_atoms, 3)
        assert coords_restored.shape == (n_confs, n_atoms, 3)

    def test_energy_weights_preserved(self, simple_entry):
        """Test that energy weights are preserved."""
        simple_entry["energy_weights"] = torch.tensor([1.0, 2.0, 3.0])
        dataset = create_weighted_dataset([simple_entry])
        entry = dataset[0]

        assert torch.allclose(entry["energy_weights"], torch.tensor([1.0, 2.0, 3.0]))

    def test_forces_weights_preserved(self, simple_entry):
        """Test that forces weights are preserved."""
        simple_entry["forces_weights"] = torch.tensor([0.5, 1.5, 2.5])
        dataset = create_weighted_dataset([simple_entry])
        entry = dataset[0]

        assert torch.allclose(entry["forces_weights"], torch.tensor([0.5, 1.5, 2.5]))

    def test_creates_dataset_from_multiple_entries(self):
        """Test that a dataset can be created from multiple entries."""
        n_confs = 2
        n_atoms = 3
        entries = [
            WeightedEntry(
                smiles=f"mol{i}",
                coords=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
                energy=torch.rand(n_confs, dtype=torch.float64),
                forces=torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
                energy_weights=torch.ones(n_confs, dtype=torch.float64),
                forces_weights=torch.ones(n_confs, dtype=torch.float64),
            )
            for i in range(3)
        ]
        dataset = create_weighted_dataset(entries)
        assert len(dataset) == 3


class TestCreateDatasetWithUniformWeights:
    """Tests for create_dataset_with_uniform_weights function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample coordinate, energy, and forces data."""
        n_confs = 4
        n_atoms = 3
        return {
            "smiles": "CCO",
            "coords": torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
            "energy": torch.rand(n_confs, dtype=torch.float64),
            "forces": torch.rand(n_confs, n_atoms, 3, dtype=torch.float64),
        }

    def test_creates_dataset_with_default_weights(self, sample_data):
        """Test that default weights are 1.0."""
        dataset = create_dataset_with_uniform_weights(**sample_data)
        entry = dataset[0]

        assert torch.all(entry["energy_weights"] == 1.0)
        assert torch.all(entry["forces_weights"] == 1.0)

    def test_creates_dataset_with_custom_energy_weight(self, sample_data):
        """Test that custom energy weight is applied."""
        sample_data["energy_weight"] = 1000.0
        dataset = create_dataset_with_uniform_weights(**sample_data)
        entry = dataset[0]

        assert torch.all(entry["energy_weights"] == 1000.0)

    def test_creates_dataset_with_custom_forces_weight(self, sample_data):
        """Test that custom forces weight is applied."""
        sample_data["forces_weight"] = 0.1
        dataset = create_dataset_with_uniform_weights(**sample_data)
        entry = dataset[0]

        assert torch.all(entry["forces_weights"] == 0.1)

    def test_nan_forces_get_zero_weight(self, sample_data):
        """Test that conformations with NaN forces get zero forces weight."""
        n_confs = 4
        n_atoms = 3
        forces = sample_data["forces"].clone()
        # Set forces for conformation 1 to NaN
        forces[1] = float("nan")
        sample_data["forces"] = forces

        dataset = create_dataset_with_uniform_weights(**sample_data)
        entry = dataset[0]

        # Conformation 1 should have zero weight, others should have 1.0
        expected_weights = torch.tensor([1.0, 0.0, 1.0, 1.0])
        assert torch.allclose(entry["forces_weights"], expected_weights)

    def test_nan_forces_with_custom_weight(self, sample_data):
        """Test that NaN forces get zero weight even with custom weight specified."""
        n_confs = 4
        forces = sample_data["forces"].clone()
        forces[2] = float("nan")
        sample_data["forces"] = forces
        sample_data["forces_weight"] = 0.5

        dataset = create_dataset_with_uniform_weights(**sample_data)
        entry = dataset[0]

        # Conformation 2 should have zero weight, others should have 0.5
        expected_weights = torch.tensor([0.5, 0.5, 0.0, 0.5])
        assert torch.allclose(entry["forces_weights"], expected_weights)


class TestMergeWeightedDatasets:
    """Tests for merge_weighted_datasets function."""

    @pytest.fixture
    def dataset_1(self):
        """Create first test dataset."""
        return create_dataset_with_uniform_weights(
            smiles="CCO",
            coords=torch.rand(3, 5, 3, dtype=torch.float64),
            energy=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            forces=torch.rand(3, 5, 3, dtype=torch.float64),
            energy_weight=1.0,
            forces_weight=1.0,
        )

    @pytest.fixture
    def dataset_2(self):
        """Create second test dataset."""
        return create_dataset_with_uniform_weights(
            smiles="CCO",
            coords=torch.rand(2, 5, 3, dtype=torch.float64),
            energy=torch.tensor([4.0, 5.0], dtype=torch.float64),
            forces=torch.rand(2, 5, 3, dtype=torch.float64),
            energy_weight=2.0,
            forces_weight=0.5,
        )

    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            merge_weighted_datasets([])

    def test_merge_single_dataset_returns_same(self, dataset_1):
        """Test that merging single dataset returns the same dataset."""
        result = merge_weighted_datasets([dataset_1])
        assert len(result) == len(dataset_1)

    def test_merge_creates_separate_entries(self, dataset_1, dataset_2):
        """Test that merging creates separate entries for each input dataset."""
        result = merge_weighted_datasets([dataset_1, dataset_2])

        # Should have 2 separate entries
        assert len(result) == 2

        # First entry should have 3 conformations
        assert len(result[0]["energy"]) == 3

        # Second entry should have 2 conformations
        assert len(result[1]["energy"]) == 2

    def test_merge_preserves_smiles(self, dataset_1, dataset_2):
        """Test that merging preserves SMILES in all entries."""
        result = merge_weighted_datasets([dataset_1, dataset_2])
        assert result[0]["smiles"] == "CCO"
        assert result[1]["smiles"] == "CCO"

    def test_merge_preserves_weights_per_entry(self, dataset_1, dataset_2):
        """Test that merging preserves weights for each entry."""
        result = merge_weighted_datasets([dataset_1, dataset_2])

        # First entry should have weight 1.0
        assert torch.all(result[0]["energy_weights"] == 1.0)
        assert torch.all(result[0]["forces_weights"] == 1.0)

        # Second entry should have weight 2.0 for energy, 0.5 for forces
        assert torch.all(result[1]["energy_weights"] == 2.0)
        assert torch.all(result[1]["forces_weights"] == 0.5)

    def test_merge_makes_energies_relative_per_entry(self, dataset_1, dataset_2):
        """Test that merged energies are made relative to minimum within each entry."""
        result = merge_weighted_datasets([dataset_1, dataset_2])

        # Minimum energy in each entry should be 0
        assert torch.isclose(
            result[0]["energy"].min(),
            torch.tensor(0.0, dtype=result[0]["energy"].dtype),
        )
        assert torch.isclose(
            result[1]["energy"].min(),
            torch.tensor(0.0, dtype=result[1]["energy"].dtype),
        )

    def test_merge_different_smiles_raises_error(self, dataset_1):
        """Test that merging datasets with different SMILES raises error."""
        dataset_different_smiles = create_dataset_with_uniform_weights(
            smiles="CC",  # Different SMILES
            coords=torch.rand(2, 5, 3, dtype=torch.float64),
            energy=torch.tensor([1.0, 2.0], dtype=torch.float64),
            forces=torch.rand(2, 5, 3, dtype=torch.float64),
        )

        with pytest.raises(ValueError, match="All datasets must have the same SMILES"):
            merge_weighted_datasets([dataset_1, dataset_different_smiles])

    def test_merge_multiple_datasets(self):
        """Test merging more than two datasets creates separate entries."""
        datasets = [
            create_dataset_with_uniform_weights(
                smiles="CCO",
                coords=torch.rand(2, 5, 3, dtype=torch.float64),
                energy=torch.rand(2, dtype=torch.float64),
                forces=torch.rand(2, 5, 3, dtype=torch.float64),
                energy_weight=float(i + 1),
            )
            for i in range(4)
        ]

        result = merge_weighted_datasets(datasets)

        # Should have 4 separate entries
        assert len(result) == 4

        # Each entry should have 2 conformations
        for entry in result:
            assert len(entry["energy"]) == 2

    def test_merge_preserves_entry_ordering(self):
        """Test that entries are in the same order as input datasets."""
        datasets = [
            create_dataset_with_uniform_weights(
                smiles="CCO",
                coords=torch.rand(2, 5, 3, dtype=torch.float64),
                energy=torch.tensor([float(i), float(i) + 1], dtype=torch.float64),
                forces=torch.rand(2, 5, 3, dtype=torch.float64),
                energy_weight=float(i + 1),
            )
            for i in range(3)
        ]

        result = merge_weighted_datasets(datasets)

        # Check that weights match expected ordering
        for i in range(3):
            assert torch.all(result[i]["energy_weights"] == float(i + 1))


class TestHasWeights:
    """Tests for has_weights function."""

    def test_weighted_dataset_has_weights(self):
        """Test that weighted dataset returns True."""
        dataset = create_dataset_with_uniform_weights(
            smiles="CCO",
            coords=torch.rand(2, 3, 3, dtype=torch.float64),
            energy=torch.rand(2, dtype=torch.float64),
            forces=torch.rand(2, 3, 3, dtype=torch.float64),
        )

        assert has_weights(dataset) is True

    def test_unweighted_dataset_has_no_weights(self):
        """Test that unweighted dataset returns False."""
        import datasets
        import pyarrow

        # Create a dataset without weight columns
        schema = pyarrow.schema(
            [
                ("smiles", pyarrow.string()),
                ("coords", pyarrow.list_(pyarrow.float64())),
                ("energy", pyarrow.list_(pyarrow.float64())),
                ("forces", pyarrow.list_(pyarrow.float64())),
            ]
        )
        table = pyarrow.Table.from_pylist(
            [
                {
                    "smiles": "CCO",
                    "coords": [0.0, 0.0, 0.0],
                    "energy": [1.0],
                    "forces": [0.0, 0.0, 0.0],
                }
            ],
            schema=schema,
        )
        dataset = datasets.Dataset(datasets.table.InMemoryTable(table))

        assert has_weights(dataset) is False

    def test_empty_dataset_has_no_weights(self):
        """Test that empty dataset returns False."""
        import datasets

        dataset = datasets.Dataset.from_dict(
            {"smiles": [], "coords": [], "energy": [], "forces": []}
        )

        assert has_weights(dataset) is False


class TestGetWeightsFromEntry:
    """Tests for get_weights_from_entry function."""

    def test_returns_weights_from_entry(self):
        """Test that weights are correctly extracted from entry."""
        entry = {
            "smiles": "CCO",
            "energy": torch.tensor([1.0, 2.0, 3.0]),
            "forces": torch.rand(3, 3, 3),
            "energy_weights": torch.tensor([1.0, 2.0, 3.0]),
            "forces_weights": torch.tensor([0.5, 1.0, 1.5]),
        }

        energy_weights, forces_weights = get_weights_from_entry(entry)

        assert torch.allclose(energy_weights, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(forces_weights, torch.tensor([0.5, 1.0, 1.5]))

    def test_returns_default_weights_when_missing(self):
        """Test that default weights are returned when not in entry."""
        entry = {
            "smiles": "CCO",
            "energy": torch.tensor([1.0, 2.0, 3.0]),
            "forces": torch.rand(3, 3, 3),
        }

        energy_weights, forces_weights = get_weights_from_entry(entry)

        expected = torch.full((3,), 1.0, dtype=torch.float64)
        assert torch.allclose(energy_weights, expected)
        assert torch.allclose(forces_weights, expected)

    def test_custom_default_weights(self):
        """Test that custom default weights can be specified."""
        entry = {
            "smiles": "CCO",
            "energy": torch.tensor([1.0, 2.0, 3.0]),
            "forces": torch.rand(3, 3, 3),
        }

        energy_weights, forces_weights = get_weights_from_entry(
            entry, default_energy_weight=1000.0, default_forces_weight=0.1
        )

        expected_energy = torch.full((3,), 1000.0, dtype=torch.float64)
        expected_forces = torch.full((3,), 0.1, dtype=torch.float64)
        assert torch.allclose(energy_weights, expected_energy)
        assert torch.allclose(forces_weights, expected_forces)

    def test_partial_weights(self):
        """Test that partial weights work correctly."""
        entry = {
            "smiles": "CCO",
            "energy": torch.tensor([1.0, 2.0, 3.0]),
            "forces": torch.rand(3, 3, 3),
            "energy_weights": torch.tensor([5.0, 5.0, 5.0]),
            # forces_weights missing
        }

        energy_weights, forces_weights = get_weights_from_entry(entry)

        assert torch.allclose(energy_weights, torch.tensor([5.0, 5.0, 5.0]))
        expected_forces = torch.full((3,), 1.0, dtype=torch.float64)
        assert torch.allclose(forces_weights, expected_forces)
