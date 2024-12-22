import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import Server
from client import ClientAPI
from model.Config import loadConfig
from model.GetDefaultModel import get_defualt_model

server = Server()
client_num = 10
for i in range(0, client_num):
    server.registerClient(client_api=ClientAPI())

for client_id, client_api in server.all_clients.items():
    other_public_keys = server.distributePublicKey(client_id=client_id)
    client_api.downLoadPublicKey(other_public_keys)

server.global_model = get_defualt_model()  # client will init the model if model == None
rounds = 4
for server.round_number in range(rounds):
    server.runRound()
