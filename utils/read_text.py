def read_text_file(file_path):
    """
    Reads the content of a text file and returns it as a string.

    Parameters:
        file_path (str): The path to the text file to be read.

    Returns:
        str: The content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except IOError as e:
        print(f"Error: An IOError occurred while reading the file: {e}")
        return None

# Example usage
# file_content = =(text[0])
# print(file_content)