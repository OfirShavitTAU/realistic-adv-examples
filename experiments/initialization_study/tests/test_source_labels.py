"""Check class-folder label inference and eligibility without running a model."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.initialization_study.configuration.loader import STUDY_DIR, get_source_label, read_config
from experiments.initialization_study.run.run_preflight import check_pairs, check_sources
from experiments.initialization_study.validation import PlanError


class SourceLabelTests(unittest.TestCase):
    """Validate class-folder IDs, path resolution, and model-prediction comparisons."""

    def setUp(self) -> None:
        self.config, self.selected = read_config(STUDY_DIR / "config.json")
        self.group = self.config["groups"]["group_1"]
        self.source = self.group["source"][0]

    def load(self, config: dict) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            read_config(path)

    def test_no_labels_needed_in_configuration(self) -> None:
        self.assertNotIn("source_labels", self.group)
        self.load(self.config)
        self.assertEqual(get_source_label(Path(self.source)), 207)

    def test_missing_or_invalid_class_folder_rejected(self) -> None:
        for folder in ('golden_retriever', '207', '207_', '-1_dog', '1000_dog', '2.5_dog', 'abc_dog'):
            with self.subTest(folder=folder):
                config = copy.deepcopy(self.config)
                config["groups"]["group_1"]["source"] = ["images/dogs/" + folder + "/source.png"]
                with self.assertRaises(PlanError):
                    self.load(config)

    def test_imagenet_edge_ids_and_leading_zeros(self) -> None:
        for folder, label in (('0_tench', 0), ('999_toilet_tissue', 999), ('0207_golden_retriever', 207)):
            self.assertEqual(get_source_label(Path('images') / folder / 'source.png'), label)

    def test_shared_source_uses_same_label_in_each_group(self) -> None:
        config = copy.deepcopy(self.config)
        config["groups"]["group_2"] = copy.deepcopy(self.group)
        self.load(config)
        self.assertEqual(get_source_label(Path(self.source)), 207)

    def test_absolute_and_relative_paths_agree(self) -> None:
        absolute = (STUDY_DIR / self.source).resolve()
        self.assertEqual(get_source_label(absolute), get_source_label(Path(self.source)))

    def test_resolved_symlink_uses_actual_class_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / '207_golden_retriever' / 'source.png'
            actual.parent.mkdir()
            actual.touch()
            alias = root / '231_collie' / 'alias.png'
            alias.parent.mkdir()
            alias.symlink_to(actual)
            self.assertEqual(get_source_label(alias), 207)

    def test_nearest_parent_must_be_the_class_folder(self) -> None:
        with self.assertRaises(PlanError):
            get_source_label(Path('images/207_golden_retriever/unlabeled/source.png'))

    def test_source_and_pair_checks_use_folder_id_not_prediction(self) -> None:
        source = (STUDY_DIR / self.source).resolve()
        initialization = (STUDY_DIR / self.group["initializations"][0]).resolve()
        groups = {"group_1": ([source], [initialization])}
        predictions = {
            source: {"path": self.source, "class_id": 207, "class_name": "golden retriever"},
            initialization: {"path": self.group["initializations"][0], "class_id": 208, "class_name": "Labrador"},
        }
        sources = check_sources(self.selected, groups, predictions)
        pairs = check_pairs(self.selected, groups, predictions)
        self.assertEqual(sources[0]["source_label_status"], "correct")
        self.assertTrue(pairs[0]["eligible"])

        predictions[source]["class_id"] = 231
        sources = check_sources(self.selected, groups, predictions)
        pairs = check_pairs(self.selected, groups, predictions)
        self.assertEqual(sources[0]["source_label_status"], "incorrect")
        self.assertEqual(pairs[0]["source_true_class_id"], 207)
        self.assertTrue(pairs[0]["changes_source_prediction"])
        self.assertFalse(pairs[0]["eligible"])


if __name__ == "__main__":
    unittest.main()
