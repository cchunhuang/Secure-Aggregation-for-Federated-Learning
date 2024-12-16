import os
import sys
import copy
import torch

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Config import loadConfig
from model.GetDefaultModel import get_defualt_model
from client.Client import Client

config_path = "./model/config.json"
config = loadConfig(config_path)

client_num = 6
dropout_num = 3
clients = [Client(client_id=i) for i in range(client_num)]

# Set the online status of each client
for idx in range(client_num):
    if idx < dropout_num:
        clients[idx].setOnlineStatus(False)
    else:
        clients[idx].setOnlineStatus(True)
    
# Upload public keys
public_keys = {client.client_id: client.uploadPublicKey() for client in clients}    

# Download public keys
for client in clients:
    client.downLoadPublicKey(public_keys)
    
# test clientUpdate
model = get_defualt_model() # client will init the model if model == None
rounds = 3
selected_clients = [client.client_id for client in clients]
for round_number in range(rounds):
    # Get model updates
    model_updates = []
    dropout_clients = []
    for client in clients:
        print(f'*****Round: {round_number}, Client: {client.client_id}*****')
        model_copy = copy.deepcopy(model)
        update = client.clientUpdate(model_copy, selected_clients, round_number)
        if update is not None:
            model_updates.append(update)
        else:
            dropout_clients.append(client.client_id)
            print(f'Client {client.client_id} is offline')
        
    # handle dropout clients
    print('*****Handling dropout clients*****')
    dropout_blinding_factors = []
    for client in clients:
        if client.online:
            model_copy = copy.deepcopy(model)
            blinding = client.dropOutHanlder(dropout_clients, model_copy, round_number)
            dropout_blinding_factors.append(blinding)
            
    blinding_sum = torch.zeros_like(dropout_blinding_factors[0])
    for blinding in dropout_blinding_factors:
        blinding_sum -= blinding
        
    # # Check blinding factors
    # print('*****Checking blinding factors*****')
    # all_blinging_sum = copy.deepcopy(blinding_sum)
    # for idx in range(client_num):
    #     if clients[idx].online:
    #         all_blinging_sum += clients[idx].blinding_factors
    # print(all_blinging_sum)
    
    # Aggregate dropout model updates
    with torch.no_grad():
        for param, blind in zip(model_updates[0].parameters(), blinding_sum):
            param.add_(blind)
    
    # Aggregate model updates
    print('*****Aggregating model updates*****')
    with torch.no_grad():
        # Aggregate online model updates
        for param, stacked_updates in zip(model.parameters(), zip(*[update.parameters() for update in model_updates])):
            stacked_updates = torch.stack(list(stacked_updates), dim=0)
            param.copy_(torch.mean(stacked_updates, dim=0))