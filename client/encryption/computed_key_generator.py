import hashlib
import random


class ComputedKeyGenerator:
    def __init__(self, prime_order_p, generator_g):
        """
        Initialize the ComputedKeyGenerator class.

        :param prime_order_p: The prime order of the cyclic group.
        :param generator_g: The generator of the cyclic group.
        """
        self.p = prime_order_p
        self.g = generator_g

    def generate_key_pair(self):
        """
        Generate a private and public key pair.

        :return: A tuple (private_key, public_key)
        """
        private_key = random.randint(1, self.p - 1)
        public_key = pow(self.g, private_key, self.p)
        return private_key, public_key

    def compute_shared_key(self, private_key, other_public_key):
        """
        Compute a shared key using the Diffie-Hellman approach.

        :param private_key: The private key of the client.
        :param other_public_key: The public key of the other client.
        :return: The computed shared key as a hash.
        """
        if not (1 <= other_public_key < self.p):
            raise ValueError("Public key must be in range [1, p-1].")

        shared_secret = pow(other_public_key, private_key, self.p)
        shared_key = hashlib.sha256(str(shared_secret).encode()).hexdigest()
        return shared_key

    def compute_all_shared_keys(self, private_key, public_keys):
        """
        Compute shared keys with all other public keys.

        :param private_key: The private key of the client.
        :param public_keys: A dictionary {client_id: public_key} of other clients.
        :return: A dictionary {client_id: shared_key}
        """
        shared_keys = {}
        for client_id, public_key in public_keys.items():
            shared_keys[client_id] = self.compute_shared_key(private_key, public_key)
        return shared_keys
