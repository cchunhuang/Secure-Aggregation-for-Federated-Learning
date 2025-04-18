import torch

def trainModel(model, train_loader, criterion, optimizer, epochs):
    """
    General training function for any model.
    
    Parameters:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        
    Returns:
        dict: A dictionary containing the training accuracy, training loss, and model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    train_accuracy = []
    train_loss = []

    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int() if labels.ndim > 1 else outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        train_accuracy.append(100 * correct / total)
        train_loss.append(running_loss / len(train_loader))
        print(f"Epoch {epoch + 1}, Accuracy: {train_accuracy[-1]:.2f}%, Loss: {train_loss[-1]:.4f}")

    return {"train_accuracy": train_accuracy, "train_loss": train_loss, "model": model}

def trainModelWithValidation(model, train_loader, valid_loader, criterion, optimizer, epochs):
    """
    General training function with validation support.
    
    Parameters:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        valid_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        
    Returns:
        dict: A dictionary containing the training accuracy, training loss, validation accuracy, validation loss, and model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_accuracy, train_loss = [], []
    valid_accuracy, valid_loss = [], []

    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int() if labels.ndim > 1 else outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        train_accuracy.append(100 * correct / total)
        train_loss.append(running_loss / len(train_loader))

        # evaluation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_running_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).int() if labels.ndim > 1 else outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

        valid_accuracy.append(100 * val_correct / val_total)
        valid_loss.append(val_running_loss / len(valid_loader))
        print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy[-1]:.2f}%, Val Acc: {valid_accuracy[-1]:.2f}%")

    return {
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
        "valid_accuracy": valid_accuracy,
        "valid_loss": valid_loss,
        "model": model,
    }

def testModel(model, test_loader):
    """
    General testing function for any model.
    
    Parameters:
        model (torch.nn.Module): Model to test.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        
    Returns:
        float: Test accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int() if labels.ndim > 1 else outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
