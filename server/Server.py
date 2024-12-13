import random
import numpy as np
from client import ClientAPI


class Server:
    def __init__(self):
        """Initialize Server
        all_clients (dict): key (int): client ID
                            value (ClientAPI): client info
        """
        self.all_clients = {}
        self.all_public_keys = {}  # all clients' public keys
        self.global_model = None
        self.updates = {}
        self.selected_clients = []  # clients (ID) selected in current round
        self.drop_out_clients = []  # drop-out clients (ID)
        self.round_number = 1
        self.current_id = 1  # ID distributed to new client
        self.prime = 2**31 - 1  # prime for compute blinding factor
        self.model_length = 1

    def registerClient(self, client_api: ClientAPI):
        """
        Register new client and upload its public key to server
        """
        client_api.setClientID(self.current_id)
        self.all_clients[client_api.getClientID()] = client_api
        self.receivePublicKey(client_api.getClientID(), client_api.uploadPublicKey())
        self.current_id += 1

    def setPrime(self, prime=2**31 - 1):
        self.prime = prime
        # if prime in Client need to be updated, write down here

    def initGlobalModel(self):
        """
        Parameters:
            model_length (int): size of model's parameter
        """
        self.global_model = np.random.rand(self.model_length)

    def receivePublicKey(self, client_id, public_key):
        """
        Receive public key from client

        Parameters:
            client_id (int)
            public_key (int)
        """
        self.all_public_keys[client_id] = public_key

    def distributePublicKey(self, client_id):
        """
        Send other clients' public key

        Parameters:
            client_id (int): current client request others public key

        Returns:
            dict: other clients' public key
                key (int): client ID
                value (int): public keys
        """
        return {k: v for k, v in self.all_public_keys.items() if k != client_id}

    def receiveClientUpdate(self, client_id, client_update):
        """
        Save update from client

        Parameters:
            client_id (int)
            client_update (torch.nn.Module): from ClientAPI.clientUpdate()
        """
        self.updates[client_id] = client_update

    def computeUpdate(self):
        """
        Compute aggregated model update (include drop-out handling)
        Returns:
            np.ndarray: Corrected model update after handling drop-outs
        """
        q_vectors = []
        for client_id in self.selected_clients:
            if client_id not in self.drop_out_clients:
                # send drop-out list to the client online
                # online client compute blinding_factors and return them
                q_vec = self.all_clients[client_id].dropOutHanlder(
                    selected_clients=self.drop_out_clients,
                    model_length=self.model_length,
                    round_number=self.round_number,
                )
                q_vectors.append(q_vec)

        # compute final q vector
        q = np.sum(q_vectors, axis=0) % self.prime

        # adjust aggregate result
        corrected_model_update = (
            np.sum(self.updates.values(), axis=0) - q
        ) % self.prime

        return corrected_model_update

    def deQuantizeUpdate(self, quantized_update, scale_factor):
        """
        Dequantize update from int to float
        Parameters:
            quantized_update (np.ndarray): quantized(aggregated) model update
            scale_factor (float)

        Returns:
            np.ndarray: (float) aggregated model
        """
        return quantized_update / scale_factor

    def updateGlobalModel(self, aggregated_update):
        """
        Parameters:
            aggregated_update (np.ndarray): from aggregateUpdate
        """
        self.global_model += aggregated_update

    def selectClients(self):
        """
        Randomly select half of clients to join current round
        """
        num_clients = max(1, len(self.all_clients) // 2)
        all_client_ids = list(self.all_clients.keys())
        if num_clients > len(all_client_ids):
            num_clients = len(all_client_ids)
        self.selected_clients = random.sample(all_client_ids, num_clients)

    def runRound(self, scale_factor):
        """
        Update selected clients and run a new round

        Parameters:
            scale_factor (float): factor used in dequantize
        """
        # Reset updates and drop-out clients list
        self.updates = {}
        self.drop_out_clients = []

        # 1. Select clients (list of client ID) for this round
        self.selectClients()

        # 2. Receive updates from selected_clients
        for client_id in self.selected_clients:
            client_api = self.all_clients[client_id]
            client_update = client_api.clientUpdate(
                model=self.global_model,
                selected_clients=self.selected_clients,
                round_number=self.round_number,
            )  # if client drop-out, receive None
            if client_update is None:
                self.drop_out_clients.append(client_id)
            else:
                self.receiveClientUpdate(client_id, client_update)

        # 3. Drop-out handling & compute model update
        corrected_model_update = self.computeUpdate()
        # 4. Dequantize update
        dequantized_update = self.deQuantizeUpdate(corrected_model_update, scale_factor)
        # 5. Update global model
        self.updateGlobalModel(dequantized_update)
        self.round_number += 1
