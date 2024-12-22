import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import Server
from client import ClientAPI
from model.Config import loadConfig
from model.GetDefaultModel import get_defualt_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(CURRENT_DIR, "result")


def main():
    server = Server()
    server.output_dir = RESULT_DIR

    client_num = 10
    rounds = 10

    for i in range(0, client_num):
        server.registerClient(client_api=ClientAPI())

    for client_id, client_api in server.all_clients.items():
        other_public_keys = server.distributePublicKey(client_id=client_id)
        client_api.downLoadPublicKey(other_public_keys)

    server.global_model = (
        get_defualt_model()
    )  # client will init the model if model == None

    start_time = time.time()

    for server.round_number in range(rounds):
        server.runRound()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Total time for {rounds} rounds: {((elapsed_time%3600)//60):.2f} mins {(elapsed_time%60):.2f} seconds"
    )


if __name__ == "__main__":
    main()
