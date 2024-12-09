import random
import hashlib


class Client:
    def __init__(self, prime=7919, generator=5, client_id=None):
        """
        Initialize the client with key generation.
        :param prime: Prime number used for key generation.
        :param generator: Generator for the group.
        :param client_id: Unique client ID.
        """
        self.prime = prime
        self.generator = generator
        self.client_id = client_id or random.randint(1, 100000)
        self.private_key = random.randint(1, self.prime - 1)
        self.public_key = pow(self.generator, self.private_key, self.prime)
        self.shared_keys = {}

    def uploadPublicKey(self):
        """
        Upload the public key.
        :return: The public key.
        """
        return self.public_key

    def downLoadPublicKey(self, keys):
        """
        Download other clients' public keys and compute shared keys.
        :param keys: A dictionary containing public keys of other clients.
        """
        self.shared_keys = {
            client_id: int(hashlib.sha256(str(pow(key, self.private_key, self.prime)).encode()).hexdigest(), 16)
            for client_id, key in keys.items()
            if client_id != self.client_id
        }

    def clientUpdate(self, model, selected_clients):
        """
        Perform model update and return the blinded model.
        :param model: Current model parameters (list).
        :param selected_clients: List of IDs of selected clients for this round.
        :return: Blinded model parameters (list).
        """
        # Simulate local training (adding random noise to the model parameters)
        updated_model = [x + random.randint(-10, 10) for x in model]

        # Generate blinding factors
        blinding_factors = [
            sum(
                ((-1) ** (self.client_id > other_id)) * self.shared_keys[other_id]
                for other_id in selected_clients if other_id != self.client_id
            )
            for _ in updated_model
        ]

        # Apply blinding to the updated model
        blinded_model = [(m + b) % self.prime for m, b in zip(updated_model, blinding_factors)]
        return blinded_model

    def dropOutHanlder(self, selected_clients):
        """
        Handle dropout clients by generating compensation blinding factors.
        :param selected_clients: List of IDs of selected clients for this round.
        :return: Compensation blinding factor sum.
        """
        blinding_factor_sum = [
            sum(
                ((-1) ** (self.client_id > other_id)) * self.shared_keys[other_id]
                for other_id in selected_clients if other_id != self.client_id
            )
            for _ in range(1)  # Simulating for one parameter as an example
        ]
        return blinding_factor_sum