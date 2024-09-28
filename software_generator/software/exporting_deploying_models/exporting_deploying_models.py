# importing necessary packages
import torch
import onnx

class ExportingDeployingModels:
    def __init__(self, best_model):
        self.best_model = best_model

    def export_model(self, format):
        if format == "PyTorch":
            torch.save(self.best_model, "best_model.pth")
        elif format == "ONNX":
            dummy_input = torch.randn(1, 3, 224, 224)  # example input shape
            torch.onnx.export(self.best_model, dummy_input, "best_model.onnx")
        else:
            print("Unsupported export format")

    def save_model_weights(self):
        torch.save(self.best_model.state_dict(), "best_model_weights.pth")

    def save_model_architecture(self):
        with open("best_model_architecture.txt", "w") as f:
            f.write(str(self.best_model))

    def save_model_hyperparameters(self, hyperparameters):
        with open("best_model_hyperparameters.txt", "w") as f:
            f.write(str(hyperparameters))