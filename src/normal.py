
def normalization_MMAB(tensor_, a = 0, b = 1, ret = True):
    """
    Normalize the given tensor to the range [a, b].

    Parameters:
    tensor (torch.Tensor): The tensor to be normalized.
    a (float): The minimum value of the desired range. Default is 0.0.
    b (float): The maximum value of the desired range. Default is 1.0.
    
    Returns:
    torch.Tensor: Normalized tensor.
    """
    x_min = tensor_.min()
    x_max = tensor_.max()

    Xmin = tensor_.min()
    Xmax = tensor_.max()
    normalized = a + (tensor_ - x_min) * (b - a) / (x_max - x_min)
    if ret:
        return normalized, \
        Xmin, Xmax
    else:
        return normalized




def denormalize_MMAB(normalized_tensor, original_min, original_max, a=0.0, b=1.0):
    """
    Denormalize the given normalized tensor to the original range [original_min, original_max].

    Parameters:
    normalized_tensor (torch.Tensor): The normalized tensor to be denormalized.
    original_min (float): The minimum value of the original data.
    original_max (float): The maximum value of the original data.
    a (float): The minimum value of the normalized range. Default is 0.0.
    b (float): The maximum value of the normalized range. Default is 1.0.

    Returns:
    torch.Tensor: Denormalized tensor.
    """
    return (normalized_tensor - a) * (original_max - original_min) / (b - a) + original_min


