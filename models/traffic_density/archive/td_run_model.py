import torch

def train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader):
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            #print("im running")
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_densenet201_model.pth')

def test(device, model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

def test_model_on_test_data(device, model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    class_correct = [0 for _ in range(len(test_loader.dataset.classes))]
    class_total = [0 for _ in range(len(test_loader.dataset.classes))]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Track accuracy for each class
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f'Test Accuracy: {overall_accuracy:.2f}%')

    # Class-wise accuracy
    for i in range(len(test_loader.dataset.classes)):
        if class_total[i] > 0:
            print(f'Accuracy of {test_loader.dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
