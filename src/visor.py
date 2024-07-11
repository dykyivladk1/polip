import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tr

import tifffile as tiff

from PIL import Image



def vis_imgs(image1, image2):
    """
    Visualizes two images side by side.
    
    Parameters:
    image1 (numpy.ndarray): The first image to visualize.
    image2 (numpy.ndarray): The second image to visualize.
    """
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions")
    
    fig, axes = plt.subplots(1, 2)
    
    axes[0].imshow(image1)
    axes[0].set_title('Image 1')
    axes[0].axis('off')  
    
    axes[1].imshow(image2)
    axes[1].set_title('Image 2')
    axes[1].axis('off')  
    plt.show()



    

def vis_imp(image_path):
    """
    Opens and displays an image using PIL.

    Args:
    - image_path (str): The file path to the image you want to display.
    """
    image = Image.open(image_path)
    image.show()




def vis_loss(losses, title='Loss over epochs', xlabel='Epoch', ylabel='Loss'):
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




def vis_gloss(gen_losses, disc_losses, title='GAN Losses over epochs', xlabel='Epoch', ylabel='Loss'):
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





def back2pil(image_tensor, show=True, denormalize=True, device = torch.device("mps")):
    """
    Converts a tensor image to a PIL image with optional denormalization and display.

    Parameters:
    image_tensor (torch.Tensor): The image tensor to convert.
    show (bool): Whether to display the image using plt.imshow(). Default is True.
    denormalize (bool): Whether to apply denormalization to the image tensor. Default is True.
    device (torch.device): The device to use for tensor operations. Default is torch.device("mps").
    
    Returns:
    PIL.Image.Image: The converted PIL image if show is False.
    """
    if image_tensor.dim() == 4:  
        image_tensor = image_tensor.squeeze(0).to(device)

    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)  # Mean for normalization
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)  # Std for normalization
        image_tensor = image_tensor.to(device) * std.to(device) + mean.to(device)  # Denormalize image
    
    else:
        image_tensor = image_tensor  

    trans = tr.ToPILImage()  
    image_pil = trans(image_tensor)

    if show:
        plt.imshow(image_pil) 
        plt.axis("off")
        plt.show()
    else:
        return image_pil  

        
def vis_tf(file_path):
    """
    Loads a TIFF file and visualizes it.

    Parameters:
    file_path (str): The path to the TIFF file.
    """
    image = tiff.imread(file_path)

    plt.imshow(image, cmap='gray')
    plt.title('TIFF Image')
    plt.axis('off') 
    plt.show()



