import hashlib

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
        # Flatten model parameters to a single list
        model_parameters = torch.cat(
            [param.data.flatten() for param in model.parameters()]
        )
        model_length = len(model_parameters)

        blinding_factors = []

        for param_index in range(model_length):
            # Compute the blinding factor for each parameter
            blinding_factor = (
                sum(
                    (-1 if client_id > other_id else 1)
                    * (
                        int(
                            hashlib.sha256(
                                f"{shared_keys[other_id]}{param_index}{round_number}".encode()
                            ).hexdigest(),
                            16,
                        )
                        % self.prime
                    )
                    for other_id in selected_clients
                    if other_id != client_id
                )
                # % self.prime
            )
            blinding_factors.append(blinding_factor)
            # # Ensure blinding factor is non-negative
            # blinding_factors.append(
            #     blinding_factor
            #     if blinding_factor >= 0
            #     else blinding_factor + self.prime
            # )

        return torch.tensor(blinding_factors, dtype=torch.float32)

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
