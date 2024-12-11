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
        self.dropout_clients = []  # drop-out clients (ID)
        self.round_number = 1
        self.current_id = 1  # ID distributed to new client
        self.prime = 1  # prime for compute blinding factor

    def registerClient(self, client_api: ClientAPI):
        """
        Register new client and upload its public key to server
        """
        client_api.setClientID(self.current_id)
        self.all_clients[client_api.getClientID()] = client_api
        self.receivePublicKey(client_api.getClientID(), client_api.uploadPublicKey())
        self.current_id += 1

    def initGlobalModel(self, model_size):
        """
        Parameters:
            model_size (int): size of model's parameter
        """
        self.global_model = np.random.rand(model_size)
        return

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

    # def distributeGlobalModel(self):
    #     """
    #     Distribute global model to selected clients

    #     Returns:
    #         dict:
    #             key (int): client ID
    #             value (int): global model
    #     """
    #     return {client_id: self.global_model for client_id in self.selected_clients}

    def receiveClientUpdate(self, client_id, client_update):
        """
        Save update from client

        Parameters:
            client_id (int)
            client_update (torch.nn.Module): from ClientAPI.clientUpdate()
        """
        self.updates[client_id] = client_update

    def dropOutHandler(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        q_vectors = []
        for client_id in self.selected_clients:
            if client_id not in self.dropout_clients:
                # send drop-out list to the client online
                # online client compute blinding_factors and return
                q_vec = self.all_clients[client_id].dropOutHanlder()
                q_vectors.append(q_vec)
        # compute final q vector
        q = np.sum(q_vectors, axis=0) % self.prime
        # adjust aggregate result
        corrected_model_update = (
            np.sum(self.updates.values(), axis=0) - q
        ) % self.prime
        return corrected_model_update

    def aggregateUpdate(self):
        """
        aggregate model updates from clients

        Returns:
            np.ndarray: aggregated model
        """
        update_matrix = np.stack(list(self.updates.values()))
        aggregated_update = np.mean(update_matrix, axis=0)
        return aggregated_update

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
        Select clients to join next round
        Update selected_clients(list of client ID)
        """
        pass

    def runRound(self, scale_factor):
        """
        Update selected clients and run a round

        Parameters:
            scale_factor (float): factor used in dequantize
        """
        self.selected_clients = self.selectClients()

        # 1. distribute global model
        # 2. receive clients update
        for client_id in self.selected_clients:
            client_api = self.all_clients[client_id]
            client_update = client_api.clientUpdate(
                model=self.global_model,
                selected_clients=self.selected_clients,
                round_number=self.round_number,
            )  # if client drop-out, receive None
            if client_update is None:
                self.dropout_clients.append(client_id)
            self.receiveClientUpdate(client_id, client_update)
        # 3. drop-out handling
        corrected_model_update = self.dropOutHandler()
        # 4. aggregate update
        aggregated_update = self.aggregateUpdate()
        # 5. dequantize update
        dequantized_update = self.deQuantizeUpdate(aggregated_update, scale_factor)
        # 6. update global model
        self.updateGlobalModel(dequantized_update)
        self.round_number += 1
