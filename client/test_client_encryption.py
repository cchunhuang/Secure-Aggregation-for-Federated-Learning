from Client import Client


def example_usage():
    prime = 7919
    generator = 5
    num_clients = 3
    initial_model = [100, 200, 300]
    round_number = 1

    # Initialize clients
    clients = [Client(prime=prime, generator=generator) for _ in range(num_clients)]

    # Clients upload their public keys
    public_keys = {client.client_id: client.uploadPublicKey() for client in clients}

    # Clients download public keys and compute shared keys
    for client in clients:
        client.downLoadPublicKey(public_keys)

    # Perform one round of federated learning
    selected_clients = [client.client_id for client in clients]
    blinded_models = [client.clientUpdate(initial_model, selected_clients, round_number) for client in clients]

    # Aggregate the models
    aggregated_model = [
        sum(blinded_model[param] for blinded_model in blinded_models) % prime
        for param in range(len(initial_model))
    ]
    print("Aggregated Model:", aggregated_model)

    # Simulate dropout handling
    dropout_client = clients[0]
    dropout_factors = dropout_client.dropOutHanlder(selected_clients, len(initial_model), round_number)
    print("Dropout Compensation Factors:", dropout_factors)


# Run the example
example_usage()
