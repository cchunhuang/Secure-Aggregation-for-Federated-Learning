import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Unit Test for ComputedKeyGenerator
import unittest
from client.encryption import ComputedKeyGenerator

class TestComputedKeyGenerator(unittest.TestCase):

    def setUp(self):
        # Example prime order and generator
        self.prime_order_p = 23  # Example small prime
        self.generator_g = 5    # Example generator
        self.key_gen = ComputedKeyGenerator(self.prime_order_p, self.generator_g)

    def test_key_pair_generation(self):
        """Test that key pairs are generated within the valid range."""
        private_key, public_key = self.key_gen.generate_key_pair()
        self.assertTrue(1 <= private_key < self.prime_order_p)
        self.assertTrue(0 <= public_key < self.prime_order_p)

    def test_shared_key_computation(self):
        """Test that shared keys are consistent between two clients."""
        # Generate key pairs for two clients
        private_key_a, public_key_a = self.key_gen.generate_key_pair()
        private_key_b, public_key_b = self.key_gen.generate_key_pair()

        # Compute shared keys
        shared_key_a = self.key_gen.compute_shared_key(private_key_a, public_key_b)
        shared_key_b = self.key_gen.compute_shared_key(private_key_b, public_key_a)

        # Shared keys should be equal
        self.assertEqual(shared_key_a, shared_key_b)

    def test_compute_all_shared_keys(self):
        """Test that shared keys are computed for all clients."""
        # Generate keys for multiple clients
        client_keys = {}
        for i in range(1, 4):
            private_key, public_key = self.key_gen.generate_key_pair()
            client_keys[i] = public_key

        # Compute shared keys for client 1
        private_key_1, public_key_1 = self.key_gen.generate_key_pair()
        shared_keys = self.key_gen.compute_all_shared_keys(private_key_1, client_keys)

        # Ensure shared keys are generated for all clients
        self.assertEqual(len(shared_keys), len(client_keys))

    def test_invalid_public_key(self):
        """Test that invalid public keys raise an error."""
        private_key, public_key = self.key_gen.generate_key_pair()
        with self.assertRaises(ValueError):
            self.key_gen.compute_shared_key(private_key, self.prime_order_p + 1)

if __name__ == "__main__":
    unittest.main()
