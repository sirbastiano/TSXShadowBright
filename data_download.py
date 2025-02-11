import requests
import os
from pathlib import Path

def download_file(url: str, save_path: str) -> None:
    """Downloads a file from a given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file should be saved.

    Raises:
        requests.HTTPError: If the request fails.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for failed requests

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded successfully: {save_path}")

# Example usage

train_url="https://zenodo.org/records/14844141/files/dataset_train.npy?download=1"
val_url="https://zenodo.org/records/14844141/files/dataset_val.npy?download=1"
test_url="https://zenodo.org/records/14844141/files/dataset_test.npy?download=1"

if __name__ == "__main__":

    os.makedirs('Data', exist_ok=True)

    for url, save_path in zip([train_url, val_url, test_url], ["./Data/dataset_train.npy", "./Data/dataset_val.npy", "./Data/dataset_test.npy"]):
        download_file(url, save_path)