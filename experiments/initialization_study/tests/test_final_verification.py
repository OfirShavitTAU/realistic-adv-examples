"""Check final classification with fake model responses and exact synthetic tensors."""

import unittest
from unittest.mock import MagicMock

import torch

from experiments.initialization_study.integration.verification import verify_final_prediction


class FinalVerificationTests(unittest.TestCase):
    """Classification success is measured independently of the researchers' success flag."""

    def model(self, class_id: int) -> MagicMock:
        model = MagicMock()
        model.num_queries = 11

        def prediction(images):
            self.assertFalse(torch.is_grad_enabled())
            model.num_queries += images.shape[0]
            return torch.tensor([class_id])

        model.predict_label.side_effect = prediction
        return model

    def test_changed_class_verifies_exact_pixels_and_one_extra_query(self) -> None:
        pixels = torch.full((3, 224, 224), 0.1234567)
        model = self.model(208)
        result = verify_final_prediction(model, pixels, 207, torch.device('cpu'))
        self.assertEqual(result.status, 'checked')
        self.assertTrue(result.attack_success)
        self.assertEqual(result.class_id, 208)
        self.assertEqual(result.class_name, 'Labrador retriever')
        self.assertEqual(result.queries, 1)
        checked = model.predict_label.call_args.args[0]
        self.assertTrue(torch.equal(checked, pixels.unsqueeze(0)))
        self.assertEqual(model.num_queries, 12)

    def test_same_class_is_verified_failure(self) -> None:
        result = verify_final_prediction(self.model(207), torch.zeros(1, 3, 224, 224), 207, torch.device('cpu'))
        self.assertEqual(result.status, 'checked')
        self.assertFalse(result.attack_success)
        self.assertEqual(result.class_name, 'golden retriever')
        self.assertEqual(result.queries, 1)

    def test_invalid_pixels_do_not_become_success_or_consume_queries(self) -> None:
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(value=value):
                model = self.model(208)
                result = verify_final_prediction(model, torch.full((3, 224, 224), value), 207, torch.device('cpu'))
                self.assertEqual(result.status, 'error')
                self.assertIsNone(result.attack_success)
                self.assertEqual(result.queries, 0)
                model.predict_label.assert_not_called()

    def test_prediction_error_records_its_cost_without_claiming_failure_or_success(self) -> None:
        model = self.model(208)

        def fail(images):
            model.num_queries += 1
            raise RuntimeError('synthetic inference error')

        model.predict_label.side_effect = fail
        result = verify_final_prediction(model, torch.zeros(3, 224, 224), 207, torch.device('cpu'))
        self.assertEqual(result.status, 'error')
        self.assertIsNone(result.attack_success)
        self.assertEqual(result.queries, 1)
        self.assertIn('synthetic inference error', result.error)


if __name__ == '__main__':
    unittest.main()
