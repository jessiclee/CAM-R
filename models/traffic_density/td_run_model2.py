from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch

def train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader, num_classes):
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # Validation step
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collecting labels, predictions, and probabilities for ROC and AUC
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())
                all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

        # Flatten arrays
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        # Compute macro precision and recall
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Compute ROC and AUC for each class
        roc_auc_dict = compute_roc_auc(all_labels, all_probs, num_classes)

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%', f'Macro-averaged Precision: {macro_precision:.4f}, Macro-averaged Recall: {macro_recall:.4f}')
        print("ROC AUC scores for each class:", roc_auc_dict)

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'weights/best_densenet201_model.pth')   # Change to suitable name

def compute_roc_auc(labels, probs, num_classes):
    # Binarize the labels for multi-class ROC computation
    binarized_labels = label_binarize(labels, classes=[i for i in range(num_classes)])
    
    roc_auc_dict = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(binarized_labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[f'Class {i}'] = round(roc_auc, 4)

    return roc_auc_dict

def test(device, model, test_loader, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collecting labels, predictions, and probabilities for ROC and AUC
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

    # Flatten arrays
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # Compute macro precision and recall
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Compute ROC and AUC for each class
    roc_auc_dict = compute_roc_auc(all_labels, all_probs, num_classes)

    # Compute overall accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%', f'Macro-averaged Precision: {macro_precision:.4f}, Macro-averaged Recall: {macro_recall:.4f}')
    print("ROC AUC scores for each class:", roc_auc_dict )

def test_model_on_test_data(device, model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    num_classes = len(test_loader.dataset.classes)
    class_names = test_loader.dataset.classes  # Extract class names
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    all_labels = []
    all_preds = []
    all_probs = []

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

            # Collecting labels, predictions, and probabilities for ROC and AUC
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

    # Flatten arrays
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # Compute macro precision and recall
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # # Compute F1 score
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Compute ROC and AUC for each class
    roc_auc_dict = compute_roc_auc(all_labels, all_probs, num_classes)

    # Compute overall accuracy
    overall_accuracy = 100 * correct / total
    print(f'Test Accuracy: {overall_accuracy:.2f}%\n')

    # Class-wise metrics
    class_precision = precision_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)
    class_recall = recall_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)

    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            roc_auc = roc_auc_dict[f'Class {i}']
            # Print all metrics for this class in one line
            print(f"Class: {class_names[i]} | "
                  f"Accuracy: {class_accuracy:.2f}% | "
                  f"Precision: {class_precision[i]:.4f} | "
                  f"Recall: {class_recall[i]:.4f} | "
                  f"ROC AUC: {roc_auc:.4f}")

    # Print macro-averaged precision, recall, and F1 score
    print(f'\nMacro-averaged Precision: {macro_precision:.4f}, Macro-averaged Recall: {macro_recall:.4f}')
    print(f'Macro-averaged F1 Score: {macro_f1:.4f}')
    print(f'Weighted F1 Score: {weighted_f1:.4f}')

def calculate_f1_score(device, model, test_loader):
    # Set the model to evaluation mode
    model.eval()
    
    all_labels = []
    all_preds = []

    # No need to track gradients during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Append true labels and predicted labels
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    # Flatten the label arrays
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Compute F1 scores
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print or return the F1 scores
    print(f'Macro-averaged F1 Score: {macro_f1:.4f}')
    print(f'Weighted F1 Score: {weighted_f1:.4f}')
    
    return macro_f1, weighted_f1