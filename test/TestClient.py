import os
import sys
import torch

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.Config import loadConfig
from client.Client import Client

config_path = "./client/config.json"
config = loadConfig(config_path)

client_num = 3
clients = [Client(client_id=i) for i in range(client_num)]

# Set the online status of each client
for client in clients:
    client.setOnlineStatus(True)
    
# Upload public keys
public_keys = {client.client_id: client.uploadPublicKey() for client in clients}    

# Download public keys
for client in clients:
    client.downLoadPublicKey(public_keys)
    
# test clientUpdate
model = None # client will init the model if model == None
rounds = 3
selected_clients = [client.client_id for client in clients]
for round_number in range(rounds):
    model_updates = []
    blinding_factors = []
    for client in clients:
        print(f'*****Round: {round_number}, Client: {client.client_id}*****')
        update, blinding = client.clientUpdate(model, selected_clients, round_number)
        model_updates.append(update)
        blinding_factors.append(blinding)
    
    # Aggregate model updates
    print('*****Aggregating model updates*****')
    model = model_updates[0]
    with torch.no_grad():
        for idx in range(1, client_num):
            for param, update in zip(model.parameters(), model_updates[idx].parameters()):
                param.add_(update)

        for param in model.parameters():
            param.div_(client_num)
            
    # check bliding factors
    print('*****Checking bliding factors*****')
    bliding_sum = torch.zeros_like(blinding_factors[0])
    for idx in range(client_num):
        bliding_sum += blinding_factors[idx]
    print(bliding_sum)