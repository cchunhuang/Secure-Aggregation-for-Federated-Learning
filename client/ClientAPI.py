from .Client import Client

class ClientAPI:
    def __init__(self):
        '''
        Initialize the client API
        '''
        self.client = Client()
    
    def uploadPublicKey(self):
        '''
        Upload the public key of the client
        
        Returns:
            int: Public key of the client
        '''
        return self.client.public_key
    
    def downLoadPublicKey(self, keys:dict):
        '''
        Download the public key of the client
        
        Parameters:
            keys (dict): Public keys of all clients
                key (int): Client ID
                value (int): Public key
                
        Returns:
            None
        '''
        self.client.downLoadPublicKey(keys)
        return
    
    def clientUpdate(self, model, selected_clients:list):
        '''
        Update the client model
        
        Parameters:
            model (torch.nn.Module): Model to update
            selected_clients (list of int): List of selected clients
            
        Returns:
            torch.nn.Module: Updated model
            or None: if the client drop out
        '''
        return self.client.clientUpdate(model, selected_clients)
    
    def dropOutHanlder(self, model, selected_clients:list):
        '''
        Handle the drop-out clients

        Parameters:
            model (torch.nn.Module): Model to update
            selected_clients (list of int): List of selected clients
            
        Returns:
            torch.nn.Module: Updated model
        '''
        return self.client.dropOutHanlder(selected_clients)