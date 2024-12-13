from .Client import Client

class ClientAPI:
    def __init__(self):
        '''
        Initialize the client API
        '''
        self.client = Client()
    
    def setOnlineStatus(self, status=None):
        """
        Set online status
        or randomly simulate the client's online status
        
        Parameters:
            status (bool, optional): if status is None, randomly set the online status
        """
        self.client.setOnlineStatus(status)
        return
    
    def setClientID(self, c_id):
        """
        Set ID got from server

        Parameters:
            c_id (int): client's new ID
        """
        self.client.client_id = c_id
        
    def getClientID(self):
        """
        return client ID
        """
        return self.client.client_id
    
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
    
    def clientUpdate(self, model, selected_clients:list, round_number):
        '''
        Update the client model
        
        Parameters:
            model (torch.nn.Module): Model to update
            selected_clients (list of int): List of selected clients
            round_number (int): Current round number
            
        Returns:
            torch.nn.Module: Updated model
            or None: if the client drop out
        '''
        return self.client.clientUpdate(model, selected_clients, round_number)
    
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