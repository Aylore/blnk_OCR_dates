from utils.read_text import read_text_file
import os
from tqdm import tqdm
from config.config import data_path

def load_data(data_path):
    """
    Load image and text files, ensuring matching pairs by name, with progress tracking.
    """

    # ## Load data
    images = [os.path.join(data_path, img) for img in os.listdir(data_path) if img.endswith(".jpg")]
    labels = [os.path.join(data_path, txt) for txt in os.listdir(data_path) if txt.endswith(".txt")]

    # Extract the file names (without extension) from both lists
    txt_file_names = set([file.rsplit('.', 1)[0] for file in labels])
    jpg_file_names = set([file.rsplit('.', 1)[0] for file in images])

    # Find common file names
    common_file_names = txt_file_names.intersection(jpg_file_names)

    # Filter the original lists to only include common elements
    common_txt_files = sorted([file for file in labels if file.rsplit('.', 1)[0] in common_file_names])
    common_jpg_files = sorted([file for file in images if file.rsplit('.', 1)[0] in common_file_names])

    # Use tqdm to show progress while reading text files
    images = common_jpg_files
    labels = [read_text_file(file) for file in tqdm(common_txt_files, desc="Loading text files")]

    return images, labels
