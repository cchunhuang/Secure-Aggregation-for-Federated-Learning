import torch
import random

from client.encryption import BlindingFactors, ComputedKeyGenerator
from client.ModelTraining import modelTraining

class Client:
    def __init__(self, prime=7919, generator=5, client_id=None, config_path=None): 
        """
        Initialize the client with key generation and blinding factors calculator.
        
        Parameters:
            prime (int): prime number for key generation
            generator (int): generator for key generation
            client_id (int, optional): client ID
        """
        self.online = True
        self.prime = prime
        self.generator = generator
        
        if client_id is None:
            self.client_id = random.randint(1, 100000)
        else:
            self.client_id = client_id
        
        self.key_generator = ComputedKeyGenerator(prime, generator)
        self.private_key, self.public_key = self.key_generator.generate_key_pair()
        self.shared_keys = {}
        self.blinding_calculator = BlindingFactors(prime=self.prime)
        
        if config_path is None:
            self.config_path = './client/config.json'
        else:
            self.config_path = config_path

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
        
        Parameters:
            keys (dict): Public keys of all clients
                key (int): Client ID
                value (int): Public key
        """
        self.shared_keys = self.key_generator.compute_all_shared_keys(private_key=self.private_key, public_keys=keys)

    def clientUpdate(self, model, selected_clients, round_number):
        """
        Perform model update and return the blinded model.
        
        Parameters:
            model (torch.nn.Module): model parameters
            selected_clients (list): list of selected clients
            round_number (int): current round number
            
        Returns:
            list: updated model with blinding factors
        """
        if self.online == False:
            return None
        
        training_result = modelTraining(self.config_path, model)
        updated_model = training_result['model']

        # Compute blinding factors using the BlindingFactors class
        blinding_factors = self.blinding_calculator.compute_blinding_factors(
            shared_keys=self.shared_keys, client_id=self.client_id,
            selected_clients=selected_clients, model=updated_model, round_number=round_number)

        # Apply blinding to the updated model
        with torch.no_grad():
            for param, blind in zip(updated_model.parameters(), blinding_factors):
                param.add_(blind)
        
        return updated_model, blinding_factors

    def dropOutHanlder(self, selected_clients, model, round_number):
        """
        Handle dropout clients by generating compensation blinding factors.
        
        Parameters:
            selected_clients (list): list of selected clients
            model_length (int): length of the model
            round_number (int): current round number
            
        Returns:
            list: compensation blinding factors
        """
        blinding_factors = self.blinding_calculator.compute_blinding_factors(
            shared_keys=self.shared_keys, client_id=self.client_id,
            selected_clients=selected_clients, model=model, round_number=round_number)
        
        return blinding_factors
