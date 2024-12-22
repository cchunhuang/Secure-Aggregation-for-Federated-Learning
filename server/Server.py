import copy
import random
import numpy as np
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        self.round_number = 0
        self.current_id = 1  # ID distributed to new client
        self.prime = 2**31 - 1  # prime for compute blinding factor
        self.output_dir = ""

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
                model_copy = copy.deepcopy(self.global_model)
                q_vec = self.all_clients[client_id].dropOutHanlder(
                    model=model_copy,
                    selected_clients=self.drop_out_clients,
                    round_number=self.round_number,
                )
                q_vectors.append(q_vec)

        # compute final q vector
        # with torch.no_grad():
        blinding_sum = torch.zeros_like(q_vectors[0])  # Initialize with zeros
        for blinding in q_vectors:
            blinding_sum -= blinding
        # q = np.sum(q_vectors, axis=0) % self.prime

        # adjust aggregate result
        corrected_model_update = list(self.updates.values())
        with torch.no_grad():
            for param, blind in zip(
                corrected_model_update[0].parameters(), blinding_sum
            ):
                param.add_(blind)
            # (
            #     np.sum(self.updates.values(), axis=0) - q
            # ) % self.prime
            aggregated_model = copy.deepcopy(corrected_model_update[0])
            for param, stacked_updates in zip(
                aggregated_model.parameters(),
                zip(*[update.parameters() for update in corrected_model_update]),
            ):
                stacked_updates = torch.stack(list(stacked_updates), dim=0)
                param.copy_(torch.mean(stacked_updates, dim=0))

        return aggregated_model

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
        # self.global_model += aggregated_update
        self.global_model = aggregated_update
        # Save model
        output_model_path = os.path.join(
            self.output_dir, str(self.round_number) + ".pth"
        )
        if output_model_path:
            torch.save(self.global_model.state_dict(), output_model_path)
            print(f"Model saved to {output_model_path}")

    def checkOnlineClients(self):
        """
        Find online clients list

        Returns:
            online_clients_id (list): online clients' ID
        """
        online_clients_id = []
        for client_id, client_api in self.all_clients.items():
            if client_api.isOnline():
                online_clients_id.append(client_id)

        return online_clients_id

    def reconnectClients(self):
        """
        Randomly (50%) recover client online status
        """
        for d_client_id in self.drop_out_clients:
            if random.random() > 0.5:
                self.all_clients[d_client_id].setOnlineStatus(True)
        return

    def selectClients(self):
        """
        Randomly select half of online clients to join current round
        """
        online_clients_id = self.checkOnlineClients()
        num_clients = max(1, len(online_clients_id) // 2)
        self.selected_clients = random.sample(online_clients_id, num_clients)

    def runRound(self):
        """
        Update selected clients and run a new round
        """
        # Recover client drop-out
        self.reconnectClients()

        # Reset updates and drop-out clients list
        self.updates = {}
        self.drop_out_clients = []

        # 1. Check clients online and select half of the online clients (list of client ID) for this round
        self.selectClients()
        print(
            f"\n*****Client selected for {self.round_number} th round*****\n{self.selected_clients}\n**********"
        )

        # 2. Receive updates from selected_clients
        for client_id in self.selected_clients:
            print(f"\n*****Round: {self.round_number}, Client: {client_id}*****")
            # To simulate client drop-out, uncomment the next if block
            if random.random() < 0.2:
                self.all_clients[
                    client_id
                ].setOnlineStatus()  # randomly set client is online or not

            model_copy = copy.deepcopy(self.global_model)
            client_api = self.all_clients[client_id]
            client_update = client_api.clientUpdate(
                model=model_copy,
                selected_clients=self.selected_clients,
                round_number=self.round_number,
            )  # if client drop-out, receive None
            if client_update is None:
                print(f"\n*****Client {client_id} is offline*****\n")
                self.drop_out_clients.append(client_id)
            else:
                self.receiveClientUpdate(client_id, client_update)

        # 3. Drop-out handling & compute model update
        corrected_model_update = self.computeUpdate()
        # 4. Dequantize update
        # dequantized_update = self.deQuantizeUpdate(corrected_model_update, scale_factor)
        # 5. Update global model
        self.updateGlobalModel(corrected_model_update)
        self.round_number += 1
