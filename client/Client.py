class Client:
    def __init__(self):
        # generate public and private key ***
        pass
    
    def uploadPublicKey(self):
        return self.public_key
    
    def downLoadPublicKey(self, keys):
        # generate computed-key (CK) ***
        pass
    
    def clientUpdate(self, model, selected_clients):
        # train model
        # generate blinding factors ***
        # return blinded model
        pass
    
    def dropOutHanlder(self, model, selected_clients):
        # for each client in selected_clients:
        #   generate blinding factors ***
        # sum all blinding factors
        # return sum
        pass