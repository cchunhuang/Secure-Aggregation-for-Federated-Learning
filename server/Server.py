import numpy as np
from client import ClientAPI


class Server:
    def __init__(self):
        self.public_keys = {}  # all clients' public keys
        self.global_model = None
        self.updates = {}
        self.selected_clients = []  # clients selected in current round
        self.dropout_clients = []

    def initGlobalModel(self, model_size):
        """
        Parameters:
            model_size (int): size of model's parameter
        """
        self.global_model = np.random.rand(model_size)
        return

    def getPublicKey(self, client_id, public_key):
        """
        receive public key from client

        Parameters:
            client_id (int)
            public_key (int): from ClientAPI.uploadPublicKey()
        """
        self.public_keys[client_id] = public_key

    def distributePublicKey(self, client_id):
        """
        send other clients' public key

        Parameters:
            client_id (int): current client request others public key

        Returns:
            dict: other clients' public key
        """
        return {k: v for k, v in self.public_keys.items() if k != client_id}

    def distributeGlobalModel(self):
        """
        distribute global model to selected clients

        Returns:
            dict:
                key (int): client id
                value (int): global model
        """
        return {client_id: self.global_model for client_id in self.selected_clients}

    def getClientUpdate(self, client_id, client_update):
        """
        get update from client

        Parameters:
            client_id (int)
            client_update (torch.nn.Module): from ClientAPI.clientUpdate()
        """
        self.updates[client_id] = client_update

    def dropOutHandler(self):
        for client in self.dropout_clients:
            if client in self.selected_clients:
                self.updates[client] = np.zeros_like(
                    self.global_model
                )  # replace by zero vec
        return self.updates

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

        Args:
            quantized_update (np.ndarray): quantized model update
            scale_factor (float)
        """
        return quantized_update / scale_factor

    def updateGlobalModel(self, aggregated_update):
        """

        Args:
            aggregated_update (np.ndarray): from aggregateUpdate
        """
        self.global_model += aggregated_update

    def runRound(self, selected_clients, scale_factor):
        """
        update selected clients and run a round

        Args:
            selected_clients (list): selected clients in this round
            scale_factor (float): factor used in dequantize
        """
        self.selected_clients = selected_clients

        # 1. distribute global model
        global_model_mapping = self.distributeGlobalModel()
        # 2. receive clients update
        for client_id in selected_clients:

            client_update = None  # call clientAPI.clientUpdate()
            self.getClientUpdate(client_id, client_update)
        # 3. drop-out handling
        self.dropOutHandler()
        # 4. aggregate update
        aggregated_update = self.aggregateUpdate()
        # 5. dequantize update
        dequantized_update = self.deQuantizeUpdate(aggregated_update, scale_factor)
        # 6. update global model
        self.updateGlobalModel(dequantized_update)
