import torch


# Saving the Model Weights
def save_model(segmentation_model, path="/content/drive/MyDrive/segmentation_model_weights/segmentation_model_weights.pth"):
    torch.save(segmentation_model.state_dict(), path)
    print(f"Model weights saved at {path}")

# Loading the Model Weights
def load_model(segmentation_model, path="/content/drive/MyDrive/segmentation_model_weights/segmentation_model_weights.pth"):
    segmentation_model.load_state_dict(torch.load(path))
    print(f"Model weights loaded from {path}")
    return segmentation_model