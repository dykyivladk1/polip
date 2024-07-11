import numpy as np
import torch
import os
import shutil

def b2t(image, tensor_ = True):
    """
    Convert a NumPy image array to a PyTorch tensor.
    
    Parameters:
    - image (np.array): Image array with shape (H, W, C)
    
    Returns:
    - tensor (torch.Tensor): Image tensor with shape (1, C, H, W)
    """
    if tensor_:
        if len(image.shape) == 3:  # Ensure image has 3 dimensions
            image = image.transpose((2, 0, 1))  # Transpose to (C, H, W)
        image = torch.tensor(image, dtype=torch.float32) # Add batch dimension
        return image
    else:
        if len(image.shape) == 3:  # Ensure image has 3 dimensions
            image = image.transpose((2, 0, 1))  # Transpose to (C, H, W)
        return image
    

def t2b(tensor, tensor_ = True):
    """
    Convert a PyTorch tensor back to a NumPy image array.
    
    Parameters:
    - tensor (torch.Tensor): Image tensor with shape (1, C, H, W)
    
    Returns:
    - image (np.array): Image array with shape (H, W, C)
    """
    if tensor_:
        image = tensor
        if len(tensor.shape) == 3:  
            image = image.permute(1, 2, 0)  # Transpose to (H, W, C)
        return image

    else:
        image = tensor
        if len(tensor.shape) == 3:  
            image = image.permute(1, 2, 0)  # Transpose to (H, W, C)
        return image
    


def get_files(path):
    """
    Retrieves the full file paths of all files in the specified directory.
    
    Parameters:
    path (str): The directory path to search for files.
    
    Returns:
    list: A list containing the full file paths of all files in the directory.
    """
    files = []
    for file in os.listdir(path):
        full_file_path = os.path.join(path, file)
        
        if os.path.isfile(full_file_path):
            files.append(full_file_path)
    return files





def move_files_in_folders(main_folder, destination_folder, var=2, copy=True):
    '''
    Move or copy files from a main folder to a destination folder based on the specified structure (VAR).
    
    Parameters:
    main_folder (str): The path to the main folder containing the files or subfolders.
    destination_folder (str): The path to the destination folder where files will be moved or copied.
    var (int, optional): Determines the structure of the main folder. Default is 2.
        - VAR 1: Assumes files are within subfolders in the main folder.
        - VAR 2: Assumes files are directly within the main folder.
    copy (bool, optional): If True, files will be copied instead of moved. Default is True.
    
    Behavior based on VAR value:
    - VAR 1:
        Assumes the main folder contains subfolders, each with files to move/copy.
        Structure before moving/copying:
        - Main folder
            - Subfolder 1
                - File 1
                - File 2
            - Subfolder 2
                - File 3
    
    - VAR 2:
        Assumes the main folder directly contains files to move/copy.
        Structure before moving/copying:
        - Main folder
            - File 1
            - File 2
            - File 3
    '''

    if var == 1:
        # Loop through each subfolder in the main folder.
        for sub_name in os.listdir(main_folder):
            subfolder_path = os.path.join(main_folder, sub_name)
            # Loop through each file in the subfolder.
            for name in os.listdir(subfolder_path):
                path = os.path.join(subfolder_path, name)
                # Ensure the destination folder exists.
                os.makedirs(destination_folder, exist_ok=True)
                # Copy or move the file to the destination folder.
                if copy:
                    shutil.copy(path, destination_folder)
                else:
                    shutil.move(path, destination_folder)
                    
    elif var == 2:
        # Loop through each file in the main folder.
        for name in os.listdir(main_folder):
            path = os.path.join(main_folder, name)
            # Copy or move the file to the destination folder.
            if copy:
                shutil.copy(path, destination_folder)
            else:
                shutil.move(path, destination_folder)




    
