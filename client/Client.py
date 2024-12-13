import random

from .encryption import BlindingFactors


class Client:
    def __init__(self, prime=7919, generator=5, client_id=None):
        """
        Initialize the client with key generation and blinding factors calculator.
        :param prime: Prime number used for key generation.
        :param generator: Generator for the group.
        :param client_id: Unique client ID.
        """
        self.online = True
        self.prime = prime
        self.generator = generator
        self.client_id = client_id or random.randint(1, 100000)
        self.private_key = random.randint(1, self.prime - 1)
        self.public_key = pow(self.generator, self.private_key, self.prime)
        self.shared_keys = {}
        self.blinding_calculator = BlindingFactors(prime=self.prime)

    def setOnlineStatus(self, status):
        """
        Set online status
        or randomly simulate the client's online status
        Parameters:
            status (bool, optional): if status is None, randomly set the online status
        """
        if status is not None:
            self.online = status
        else:
            self.online = random.choice([True, False])
        
    def uploadPublicKey(self):
        """
        Upload the public key.
        """
        return self.public_key

    def downLoadPublicKey(self, keys):
        """
        Download other clients' public keys and compute shared keys.
        :param keys: Dictionary containing public keys of other clients.
        """
        self.shared_keys = {
            client_id: pow(key, self.private_key, self.prime)
            for client_id, key in keys.items()
                if client_id != self.client_id
        }

    def clientUpdate(self, model, selected_clients, round_number):
        """
        Perform model update and return the blinded model.
        :param model: Current model parameters (list).
        :param selected_clients: List of IDs of selected clients for this round.
        :param round_number: Current round number.
        :return: Blinded model parameters (list).
        """
        # Simulate local training (adding random noise to the model parameters)
        updated_model = [x + random.randint(-10, 10) for x in model]

        # Compute blinding factors using the BlindingFactors class
        blinding_factors = self.blinding_calculator.compute_blinding_factors(
            shared_keys=self.shared_keys,
            client_id=self.client_id,
            selected_clients=selected_clients,
            model_length=len(model),
            round_number=round_number,
        )

        # Apply blinding to the updated model
        blinded_model = [
            (m + b) % self.prime for m, b in zip(updated_model, blinding_factors)
        ]
        return blinded_model

    def dropOutHanlder(self, selected_clients, model_length, round_number):
        """
        Handle dropout clients by generating compensation blinding factors.
        :param selected_clients: List of IDs of selected clients for this round.
        :param model_length: Number of parameters in the model.
        :param round_number: Current round number.
        :return: Compensation blinding factor sum.
        """
        blinding_factors = self.blinding_calculator.compute_blinding_factors(
            shared_keys=self.shared_keys,
            client_id=self.client_id,
            selected_clients=selected_clients,
            model_length=model_length,
            round_number=round_number,
        )
        return blinding_factors
