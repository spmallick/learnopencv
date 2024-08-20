import os
import subprocess
import sys

def run_command(command):
    """Runs a shell command."""
    result = subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    return result

def install_packages():
    """Installs the required Python packages."""
    try:
        print("Installing required packages using pip...")
        run_command("pip install qdrant-client pandas -q")
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def create_data_directory():
    """Creates a data directory to store the MovieLens dataset."""
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory.")

def download_and_extract_data():
    """Downloads and extracts the MovieLens dataset."""
    try:
        print("Downloading MovieLens dataset...")
        run_command("wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -q")
        print("Extracting MovieLens dataset...")
        run_command("unzip -q ml-latest-small.zip -d data")
        print("Dataset downloaded and extracted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading or extracting dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_packages()
    create_data_directory()
    download_and_extract_data()
