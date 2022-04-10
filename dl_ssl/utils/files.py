# Standard imports
import os

# Importing this package for package file manipulation
import dl_ssl

def make_dir(folder_path):
    if not os.path.exists(folder_path):
        print(f"Making directory: {folder_path}")
        os.makedirs(folder_path)
    else:
        print(f"Directory already exists: {folder_path}")

def get_path_in_package(rel_path):
    package_path = dl_ssl.__path__[0]
    return os.path.join(package_path, rel_path)