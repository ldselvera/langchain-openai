# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np

class ResultsVisualization:
    def __init__(self, best_architecture, best_hyperparameters, evaluation_metrics):
        self.best_architecture = best_architecture
        self.best_hyperparameters = best_hyperparameters
        self.evaluation_metrics = evaluation_metrics
    
    def display_summary(self):
        print("Best Architecture: ", self.best_architecture)
        print("Best Hyperparameters: ", self.best_hyperparameters)

    def display_performance(self, model_configurations, evaluation_results):
        for i, configuration in enumerate(model_configurations):
            print("Model Configuration ", i+1)
            print("Architecture: ", configuration['architecture'])
            print("Hyperparameters: ", configuration['hyperparameters'])
            print("Evaluation Results: ", evaluation_results[i])

    def visualize_training_progress(self, training_loss, validation_loss):
        epochs = np.arange(1, len(training_loss) + 1)
        plt.plot(epochs, training_loss, label="Training Loss")
        plt.plot(epochs, validation_loss, label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def visualize_confusion_matrices(self, confusion_matrices):
        for i, matrix in enumerate(confusion_matrices):
            plt.figure(i+1)
            plt.imshow(matrix, cmap='Blues')
            plt.colorbar()
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - Model ' + str(i+1))
            plt.show()