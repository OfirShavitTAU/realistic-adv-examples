"""Check original-HSJA delegation using mocks instead of model computation."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from foolbox.distances import l2, linf

from experiments.initialization_study.configuration.loader import STUDY_DIR, read_config
from experiments.initialization_study.integration import hsja
from experiments.initialization_study.validation import PlanError
from src.attacks.hsja import HSJA


class HSJAAdapterTests(unittest.TestCase):
    """Verify task settings, tensor layout, untargeted mode, and device compatibility."""

    def setUp(self) -> None:
        config, _ = read_config(STUDY_DIR / "config.json")
        self.settings = config["attack"]
        self.settings["norms"] = ["l2", "linf"]

    def image(self) -> MagicMock:
        image = MagicMock(spec=torch.Tensor)
        image.device = torch.device("cuda:0")
        image.shape = torch.Size((1, 3, 224, 224))
        image.dtype = torch.float32
        return image

    def test_original_constructor_and_norms(self) -> None:
        for norm, distance in (("l2", l2), ("linf", linf)):
            attack = hsja.create_hsja(self.settings, norm)
            self.assertIs(type(attack), HSJA)
            self.assertIs(attack.distance, distance)
            self.assertEqual(attack.queries_limit, self.settings["max_queries"])
            self.assertEqual(attack.unsafe_queries_limit, self.settings["max_unsafe_queries"])
            self.assertEqual(attack.num_iterations, self.settings["max_iterations"])
            self.assertFalse(attack.discrete)

    def test_image_and_random_delegation(self) -> None:
        source = self.image()
        label = self.image()
        label.shape = torch.Size((1,))
        label.dtype = torch.long
        model = MagicMock()
        parameter = MagicMock()
        parameter.device = source.device
        model.parameters.side_effect = lambda: iter([parameter])

        for initialization in (None, self.image()):
            # Replace only the original algorithm body; its constructor is real.
            with patch.object(HSJA, "hsja", return_value=object()) as original_call:
                result = hsja.run_hsja(model, source, label, self.settings, "l2", initialization)
                self.assertIs(result, original_call.return_value)
                arguments = original_call.call_args.kwargs
                self.assertIs(arguments["model"], model)
                self.assertIs(arguments["sample"], source)
                self.assertIs(arguments["original_label"], label)
                self.assertIsNone(arguments["target_label"])
                expected_image = initialization[0] if initialization is not None else None
                self.assertIs(arguments["target_image"], expected_image)
                self.assertEqual(arguments["num_iterations"], self.settings["max_iterations"])
                self.assertIs(arguments["distance"], l2)
                model.predict_label.assert_not_called()

    def test_cpu_tensors_delegate_without_changing_original_attack(self) -> None:
        source = self.image()
        source.device = torch.device("cpu")
        label = self.image()
        label.device = source.device
        label.shape = torch.Size((1,))
        label.dtype = torch.long
        model = MagicMock()
        parameter = MagicMock()
        parameter.device = source.device
        model.parameters.return_value = iter([parameter])

        with patch.object(HSJA, "hsja", return_value=object()) as original_call:
            result = hsja.run_hsja(model, source, label, self.settings, "l2")
            self.assertIs(result, original_call.return_value)
            self.assertIs(original_call.call_args.kwargs["sample"], source)
            self.assertIsNone(original_call.call_args.kwargs["target_label"])

    def test_model_device_mismatch_rejected_before_attack(self) -> None:
        source = self.image()
        source.device = torch.device("cpu")
        model = MagicMock()
        parameter = MagicMock()
        parameter.device = torch.device("cuda:0")
        model.parameters.return_value = iter([parameter])

        with patch.object(hsja, "create_hsja") as create:
            with self.assertRaises(PlanError):
                hsja.run_hsja(model, source, MagicMock(), self.settings, "l2")
            create.assert_not_called()

    def test_invalid_initialization_rejected(self) -> None:
        source = self.image()
        label = self.image()
        label.shape = torch.Size((1,))
        label.dtype = torch.long
        model = MagicMock()
        parameter = MagicMock()
        parameter.device = source.device
        model.parameters.side_effect = lambda: iter([parameter])
        initialization = self.image()
        initialization.shape = torch.Size((3, 224, 224))

        with patch.object(hsja, "create_hsja") as create:
            with self.assertRaises(PlanError):
                hsja.run_hsja(model, source, label, self.settings, "l2", initialization)
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
