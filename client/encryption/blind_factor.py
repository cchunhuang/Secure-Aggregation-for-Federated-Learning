import hashlib
from concurrent.futures import ThreadPoolExecutor

import torch


class BlindingFactors:
    def __init__(self, prime=7919):
        """
        Initialize the BlindingFactors class.
        :param prime: Prime number for modular arithmetic.
        """
        self.prime = prime

    def compute_blinding_factors(
        self, shared_keys, client_id, selected_clients, model, round_number
    ):
        """
        Compute the vector of blinding factors for the client.
        :param shared_keys: Dictionary of shared keys {other_client_id: CK_k,n}.
        :param client_id: The current client's ID.
        :param selected_clients: List of IDs of selected clients for this round.
        :param model: PyTorch model (nn.Module) with parameters to blind.
        :param round_number: Current round number t.
        :return: A list representing the vector of blinding factors.
        """
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Flatten model parameters to a single list
        model_parameters = torch.cat(
            [param.data.flatten() for param in model.parameters()]
        )
        model_length = len(model_parameters)

        # Filter selected_clients to exclude the current client
        selected_clients = [
            other_id for other_id in selected_clients if other_id != client_id
        ]

        # Precompute signs for selected clients
        signs = torch.tensor(
            [-1 if client_id > other_id else 1 for other_id in selected_clients],
            dtype=torch.float32,
            device=device,
        )

        # Prepare hash inputs as a list for concurrent processing
        keys_array = [shared_keys[other_id] for other_id in selected_clients]

        # Function to compute a single hash value
        def compute_hash(shared_key, param_index):
            hash_input = f"{shared_key}{param_index}{round_number}".encode()
            return int(hashlib.sha256(hash_input).hexdigest(), 16) % self.prime

        # Function to compute hashes for one client
        def compute_client_hashes(shared_key):
            return [
                compute_hash(shared_key, param_index)
                for param_index in range(model_length)
            ]

        # Compute all hashes concurrently
        with ThreadPoolExecutor() as executor:
            all_hashes = list(executor.map(compute_client_hashes, keys_array))

        # Convert hashes to a PyTorch tensor
        hash_tensor = torch.tensor(all_hashes, dtype=torch.float32, device=device)

        # Apply signs and sum over clients
        blinding_factors = torch.sum(signs[:, None] * hash_tensor, dim=0)

        return blinding_factors

    def apply_blinding(self, model, blinding_factors):
        """
        Apply blinding factors to the PyTorch model.
        :param model: PyTorch model (nn.Module) to apply blinding to.
        :param blinding_factors: Tensor of blinding factors to add to the model parameters.
        :return: None (applies in-place modifications to the model).
        """
        model_parameters = torch.cat(
            [param.data.flatten() for param in model.parameters()]
        )
        blinded_parameters = model_parameters + blinding_factors % self.prime

        # Reassign blinded parameters back to the model
        current_index = 0
        for param in model.parameters():
            param_length = param.numel()
            param.data = blinded_parameters[
                current_index : current_index + param_length
            ].view_as(param)
            current_index += param_length
