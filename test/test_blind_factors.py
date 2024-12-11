import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import unittest
import torch
from client.encryption import BlindingFactors

class TestBlindingFactors(unittest.TestCase):
    def setUp(self):
        # Example PyTorch model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.fc2(x)
                return x

        self.model = SimpleModel()

        # Dummy shared keys and selected clients
        self.shared_keys = {1: "key1", 2: "key2", 3: "key3"}
        self.selected_clients = [1, 2, 3]
        self.client_id = 2
        self.round_number = 1
        self.blinding_calculator = BlindingFactors()

    def test_compute_blinding_factors_length(self):
        """
        Test to ensure the length of the blinding factors matches the total number of model parameters.
        """
        blinding_factors = self.blinding_calculator.compute_blinding_factors(
            shared_keys=self.shared_keys,
            client_id=self.client_id,
            selected_clients=self.selected_clients,
            model=self.model,
            round_number=self.round_number,
        )
        # Ensure the length matches the total number of model parameters
        total_params = sum(param.numel() for param in self.model.parameters())
        self.assertEqual(len(blinding_factors), total_params)

    def test_apply_blinding(self):
        """
        Test to verify that applying blinding factors modifies the model parameters
        and ensures that parameter shapes remain consistent.
        """
        blinding_factors = self.blinding_calculator.compute_blinding_factors(
            shared_keys=self.shared_keys,
            client_id=self.client_id,
            selected_clients=self.selected_clients,
            model=self.model,
            round_number=self.round_number,
        )

        # Capture original parameters
        original_params = [param.data.clone() for param in self.model.parameters()]

        # Apply blinding
        self.blinding_calculator.apply_blinding(self.model, blinding_factors)

        # Check that parameters are modified and shapes remain the same
        for original, param in zip(original_params, self.model.parameters()):
            # Ensure the shapes remain the same
            self.assertEqual(original.shape, param.data.shape, "Parameter shape changed after blinding.")
            # Ensure the values are modified
            self.assertFalse(torch.equal(original, param.data), "Parameter values did not change after blinding.")


if __name__ == "__main__":
    unittest.main()
