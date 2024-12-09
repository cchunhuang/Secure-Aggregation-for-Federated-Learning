import hashlib


class BlindingFactors:
    def __init__(self, prime=7919):
        """
        Initialize the BlindingFactors class.
        :param prime: Prime number for modular arithmetic.
        """
        self.prime = prime

    def compute_blinding_factors(self, shared_keys, client_id, selected_clients, model_length, round_number):
        """
        Compute the vector of blinding factors for the client.
        :param shared_keys: Dictionary of shared keys {other_client_id: CK_k,n}.
        :param client_id: The current client's ID.
        :param selected_clients: List of IDs of selected clients for this round.
        :param model_length: Length of the model (number of parameters).
        :param round_number: Current round number t.
        :return: A list representing the vector of blinding factors.
        """
        blinding_factors = []

        for param_index in range(model_length):
            # Compute the blinding factor for each parameter
            blinding_factor = sum(
                ((-1) ** (client_id > other_id)) *
                int(hashlib.sha256(f"{shared_keys[other_id]}||{param_index}||{round_number}".encode()).hexdigest(), 16)
                for other_id in selected_clients if other_id != client_id
            ) % self.prime
            blinding_factors.append(blinding_factor)

        return blinding_factors
