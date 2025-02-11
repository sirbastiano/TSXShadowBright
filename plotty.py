
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_batch(batch):
    images, masks = batch
    
    # Display the first image and its corresponding mask in the batch
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # Display the image
    axes[0].imshow(images[0, 0], cmap='gray')  # Assuming the image is in the first channel
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Display the mask
    axes[1].imshow(masks[0], cmap='viridis')  # Assuming the mask is in the first channel
    axes[1].set_title('Mask')
    axes[1].axis('off')

    # Show the plot
    plt.show()
    print(np.unique(masks[0]))  # Display unique values in the mask
    
    
def plot_csv(csv_file):
    # Load the CSV file
    file_path = csv_file
    df = pd.read_csv(file_path)

    # Drop rows with NaN values in 'train_loss_epoch' and 'val_loss' columns
    df_train_loss = df.dropna(subset=['train_loss_epoch'])
    df_val_loss = df.dropna(subset=['val_loss'])

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(df_train_loss['epoch'], df_train_loss['train_loss_epoch'], label='Train Loss')
    plt.plot(df_val_loss['epoch'], df_val_loss['val_loss'], label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlim([0,20])
    plt.legend()

    # Show the plot
    plt.show()
    return df