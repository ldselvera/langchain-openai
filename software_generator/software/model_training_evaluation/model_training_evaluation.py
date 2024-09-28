import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

class ModelTrainingEvaluation:
    def __init__(self, dataset, search_space, evaluation_metrics):
        self.dataset = dataset
        self.search_space = search_space
        self.evaluation_metrics = evaluation_metrics

    def split_dataset(self, train_ratio, val_ratio, test_ratio):
        # Split the dataset into training, validation, and test sets
        train_size = int(len(self.dataset) * train_ratio)
        val_size = int(len(self.dataset) * val_ratio)
        test_size = len(self.dataset) - train_size - val_size

        train_set, val_set, test_set = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        return train_set, val_set, test_set

    def train_model(self, train_set, val_set, architecture, hyperparameters):
        # Train the model using PyTorch
        model = architecture.to(device)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=hyperparameters['learning_rate'])
        scheduler = StepLR(optimizer, step_size=hyperparameters['step_size'], gamma=hyperparameters['gamma'])

        train_loader = DataLoader(train_set, batch_size=hyperparameters['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=hyperparameters['batch_size'])

        for epoch in range(hyperparameters['num_epochs']):
            # Training loop
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation loop
            model.eval()
            total_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss = total_loss / len(val_loader)
            val_accuracy = correct / total

            # Update the learning rate
            scheduler.step()

            # Display the training progress
            print(f"Epoch {epoch+1}/{hyperparameters['num_epochs']}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return model

    def evaluate_model(self, model, test_set):
        # Evaluate the trained model using the specified evaluation metrics
        test_loader = DataLoader(test_set, batch_size=32)

        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = total_loss / len(test_loader)
        test_accuracy = correct / total

        # Calculate the specified evaluation metrics
        evaluation_metrics = self.evaluation_metrics.calculate_metrics(predicted, labels)

        # Display the model evaluation results
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print("Evaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")

        return evaluation_metrics

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ...

user_interface = UserInterface()
search_space = user_interface.define_search_space()
evaluation_metrics = user_interface.choose_evaluation_metrics()

model_training_evaluation = ModelTrainingEvaluation(dataset, search_space, evaluation_metrics)
train_set, val_set, test_set = model_training_evaluation.split_dataset(0.8, 0.1, 0.1)
architecture = user_interface.generate_architecture()
hyperparameters = user_interface.generate_hyperparameters()

trained_model = model_training_evaluation.train_model(train_set, val_set, architecture, hyperparameters)
evaluation_metrics = model_training_evaluation.evaluate_model(trained_model, test_set)
