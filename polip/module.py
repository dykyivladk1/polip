import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
import matplotlib.patches as patches


import torchvision.transforms as transforms


from PIL import Image
import numpy as np
import sys
import os



#transforms
def get_rgb_transform(resize=(256, 256), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Create a torchvision transforms composition for RGB images.

    Args:
    - resize (tuple): Target size for the Resize transform, as (width, height).
    - mean (list): Mean for each channel for normalization.
    - std (list): Standard deviation for each channel for normalization.

    Returns:
    - A torchvision.transforms.Compose object with the specified transformations.
    """
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_gray_transform(resize=(256, 256), mean=[0.5], std=[0.5]):
    """
    Create a torchvision transforms composition for grayscale images.

    Args:
    - resize (tuple): Target size for the Resize transform, as (width, height).
    - mean (list): Mean for normalization (single value in a list for grayscale).
    - std (list): Standard deviation for normalization (single value in a list for grayscale).

    Returns:
    - A torchvision.transforms.Compose object with the specified transformations.
    """
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


#weights init
def weights_init(m):
    """
    Custom weights initializer for the network.

    Args:
    - m (torch.nn.Module): PyTorch layer to initialize.
    """
    if isinstance(m, nn.Conv2d):
        # Initialize Conv2d layers with a normal distribution with mean=0.0 and std=0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm2d layers' weights with a normal distribution with mean=1.0 and std=0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # Initialize BatchNorm2d layers' biases to 0
        nn.init.constant_(m.bias.data, 0)


#dataset custom
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    

#print progress bar
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print() 
        

#denormalize
def denormalize_tensor(tensor, mean, std):
    """
    Denormalizes a tensor using the provided mean and std.

    Args:
    - tensor (torch.Tensor): The image tensor to denormalize.
    - mean (list or tuple): The mean used for normalization (per channel).
    - std (list or tuple): The standard deviation used for normalization (per channel).

    Returns:
    - torch.Tensor: A denormalized image tensor.
    """
    if tensor.is_cuda:
        mean = torch.tensor(mean, device=tensor.device)
        std = torch.tensor(std, device=tensor.device)
    else:
        mean = torch.tensor(mean)
        std = torch.tensor(std)

    denormalized = tensor.clone()
    for t, m, s in zip(denormalized, mean, std):
        t.mul_(s).add_(m) 
    return denormalized



def denormalize_numpy(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes an image using the provided mean and std.

    Args:
    - image (numpy.ndarray or torch.Tensor): The normalized image to denormalize.
    - mean (list or tuple): The mean used for normalization (per channel).
    - std (list or tuple): The standard deviation used for normalization (per channel).

    Returns:
    - numpy.ndarray: A denormalized image.
    """
    mean = np.array(mean)
    std = np.array(std)
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    denormalized = image * std + mean 
    denormalized = np.clip(denormalized, 0, 1)  
    return denormalized



def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std + mean).clip(0, 1)
    return image

#display image
def display_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Displays an image from a normalized tensor.

    Args:
    - tensor (torch.Tensor): The normalized image tensor to display.
    - mean (list): The mean used for normalization (per channel).
    - std (list): The standard deviation used for normalization (per channel).
    """
    tensor = denormalize(tensor, mean, std)
    img = tensor.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')  
    plt.show()
    


def visualize_model(model, max_layers_per_subplot=10):
    """
    Visualizes the architecture of a PyTorch model by plotting its layers as colored blocks.

    Parameters:
    - model: The PyTorch model to visualize. The model should be an instance of a class derived from torch.nn.Module.
    - max_layers_per_subplot: The maximum number of layers to display in a single subplot. Default is 10.

    The function parses the model's layers, organizes them into a readable format, and then uses matplotlib to draw a diagram with each layer represented as a block. The layers are arranged vertically, with connections indicated by lines between blocks. This visualization helps in understanding the model's structure at a glance.
    """

    def convert_layers_to_architecture(layers_dict):
        architecture = []
        for name, layer in layers_dict.items():
            if name:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else name
                if not any(parent_name in d.get('parent', '') for d in architecture):
                    layer_info = {"layer": str(layer), "parent": parent_name}
                    architecture.append(layer_info)
        return architecture

    layers_dict = {name: module for name, module in model.named_modules() if name}
    architecture = convert_layers_to_architecture(layers_dict)

    # Determine the number of subplots needed
    num_layers = len(architecture)
    num_subplots = (num_layers + max_layers_per_subplot - 1) // max_layers_per_subplot

    def draw_block(ax, center_x, center_y, width, height, text, color='lightgrey'):
        block = patches.Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                                  linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(block)
        ax.text(center_x, center_y, text, ha='center', va='center', fontsize=8, wrap=True)
        return center_y - height / 2

    def draw_architecture(ax, architecture_subset):
        layer_width = 0.8
        layer_height = 0.1
        vertical_spacing = 0.1
        current_y = 0

        for i, layer in enumerate(architecture_subset):
            layer_text = layer["layer"]
            if len(layer_text) > 30:
                layer_text = '\n'.join([layer_text[:30], layer_text[30:]])
            color = 'lightblue' if 'block' in layer['parent'].lower() else 'lightgrey'
            bottom_y = draw_block(ax, 0.5, current_y, layer_width, layer_height, layer_text, color=color)

            if i < len(architecture_subset) - 1:
                next_layer_top_y = bottom_y - vertical_spacing + layer_height / 2
                ax.plot([0.5, 0.5], [bottom_y, next_layer_top_y], 'k-')
                current_y = next_layer_top_y - layer_height / 2

        ax.set_xlim(0, 1)
        ax.set_ylim(bottom_y - 0.1, 0.1)
        ax.axis('off')

    fig, axs = plt.subplots(num_subplots, 1, figsize=(6, 8 * num_subplots))

    if num_subplots == 1:
        axs = [axs]  

    for i, ax in enumerate(axs):
        start_idx = i * max_layers_per_subplot
        end_idx = start_idx + max_layers_per_subplot
        architecture_subset = architecture[start_idx:end_idx]
        draw_architecture(ax, architecture_subset)

    plt.show()
    
    
def decider(preferred_device=None):
    """
    Decides which device to use based on availability and preference.

    Args:
    - preferred_device (str, optional): The preferred device to use. Options are 'mps', 'cpu', 'gpu'. 
                                         If not specified, the function will automatically choose the best available device.

    Returns:
    - torch.device: The selected PyTorch device.
    """
    if preferred_device is not None:
        preferred_device = preferred_device.lower()
        if preferred_device == 'mps' and torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders) for acceleration.")
            return torch.device('mps')
        elif preferred_device == 'gpu' and torch.cuda.is_available():
            print("Using GPU for acceleration.")
            return torch.device('cuda')
        elif preferred_device == 'cpu':
            print("Using CPU.")
            return torch.device('cpu')
        else:
            print(f"Preferred device '{preferred_device}' not available. Choosing automatically.")

    if torch.backends.mps.is_available():
        print("Automatically selected MPS (Metal Performance Shaders) for acceleration.")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("Automatically selected GPU for acceleration.")
        return torch.device('cuda')
    else:
        print("GPU and MPS not available. Using CPU.")
        return torch.device('cpu')
    

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Calculates the gradient penalty for enforcing Lipschitz constraint in WGAN-GP.

    Args:
    - critic (torch.nn.Module): The critic (or discriminator) network.
    - real (torch.Tensor): Real images tensor.
    - fake (torch.Tensor): Fake images tensor generated by the generator.
    - alpha (float): The alpha parameter for mixing real and fake images.
    - train_step (int): Current training step (used if the critic's behavior changes over training).
    - device (str, optional): The device tensors are stored on ("cpu", "cuda", etc.).

    Returns:
    - torch.Tensor: The calculated gradient penalty.
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1) 
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty



def visualize_image_pil(image_path):
    """
    Opens and displays an image using PIL.

    Args:
    - image_path (str): The file path to the image you want to display.
    """
    image = Image.open(image_path)
    image.show()
    
def count_files_in_folder(folder_path):
    """
    Counts the number of files in the specified folder.

    Args:
    - folder_path (str): The path to the folder whose files you want to count.

    Returns:
    - int: The number of files in the folder.
    """
    all_entries = os.listdir(folder_path)
    file_count = sum(os.path.isfile(os.path.join(folder_path, entry)) for entry in all_entries)
    return file_count



def plot_loss(losses, title='Loss over epochs', xlabel='Epoch', ylabel='Loss'):
    """
    Plots the loss over epochs or iterations.

    Args:
    - losses (list or array): A list or array containing loss values.
    - title (str): The title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_gan_losses(gen_losses, disc_losses, title='GAN Losses over epochs', xlabel='Epoch', ylabel='Loss'):
    """
    Plots the losses of the generator and discriminator over epochs or iterations.

    Args:
    - gen_losses (list or array): A list or array containing generator loss values.
    - disc_losses (list or array): A list or array containing discriminator loss values.
    - title (str): The title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()





def save_model(model, folder_path, filename='model.pt'):
    """
    Saves a PyTorch model to the specified folder.

    Args:
    - model (torch.nn.Module or tuple): The model or a tuple of models to save.
    - folder_path (str): The folder path where the model will be saved.
    - filename (str, optional): The filename for the saved model. Default is 'model.pt'.
    """
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, filename)

    torch.save(model, file_path)