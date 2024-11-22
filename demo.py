import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

def myConv2(A, B, param=None):
    """
    Performs 2D convolution between two matrices A and B.
    param: If set to 'same', the output will have the same dimensions as A.
    """
    # Validate input types
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    
    # Dimensions of the input matrices
    M, N = A.shape
    m, n = B.shape
    
    # Handle padding for 'same' output
    if param == 'same':
        pad_h = (m - 1) // 2
        pad_w = (n - 1) // 2
        A_padded = np.pad(A, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        y_out, x_out = M, N
    else:
        A_padded = A
        y_out = M - m + 1
        x_out = N - n + 1
    
    # Initialize the result matrix
    result = np.zeros((y_out, x_out))
    
    # Perform convolution
    for i in range(y_out):
        for j in range(x_out):
            result[i, j] = np.sum(A_padded[i:i+m, j:j+n] * B)
            
    return result

def myColorToGray(A, param=None):
    """
    Converts a color image (3D array) into grayscale.
    Uses BT.601 standard weights for the RGB channels.
    """
    # Ensure the input is a color image
    if len(A.shape) != 3:
        raise ValueError("Input must be a color image")
        
    # Weights for RGB channels
    weights = np.array([0.299, 0.587, 0.114])
    
    # Compute the grayscale image
    gray = np.dot(A[..., :3], weights)
    return gray

def myImNoise(A, param='gaussian', mean=0, std=0.1, amount=0.05):
    """
    Adds noise to an image.
    param: Type of noise ('gaussian' or 'saltandpepper').
    - 'gaussian': Adds Gaussian noise with specified mean and standard deviation.
    - 'saltandpepper': Adds salt-and-pepper noise with a given amount.
    """
    # Create a copy of the input to preserve the original image
    noisy = np.copy(A)
    
    if param == 'gaussian':
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, A.shape)
        noisy = A + noise
        # Clip values to stay within valid range [0, 1]
        noisy = np.clip(noisy, 0, 1)
    
    elif param == 'saltandpepper':
        # Generate salt noise
        mask = np.random.random(A.shape) < amount
        noisy[mask] = 1  # Salt
        # Generate pepper noise
        mask = np.random.random(A.shape) < amount
        noisy[mask] = 0  # Pepper
    
    return noisy

def myImFilter(A, param='mean', kernel_size=3):
    """
    Applies a filter to an image.
    param: Type of filter ('mean' or 'median').
    kernel_size: Size of the filtering window (must be odd).
    """
    # Kernel size must be odd for symmetry
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
        
    # Padding size
    pad = kernel_size // 2
    
    # Reflect padding to handle borders
    A_padded = np.pad(A, pad, mode='reflect')
    
    # Initialize the result matrix
    result = np.zeros_like(A)
    
    # Apply the filter
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # Extract the kernel window
            window = A_padded[i:i+kernel_size, j:j+kernel_size]
            if param == 'mean':
                result[i, j] = np.mean(window)
            elif param == 'median':
                result[i, j] = np.median(window)
                
    return result

def myEdgeDetection(A, param='sobel'):
    """
    Detects edges in an image.
    param: Type of edge detection ('sobel', 'prewitt', or 'laplacian').
    """
    # Convert to grayscale if input is a color image
    if len(A.shape) > 2:
        A = myColorToGray(A)
        
    if param == 'sobel':
        # Sobel kernels for edge detection
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
        kernel_y = kernel_x.T
        
    elif param == 'prewitt':
        # Prewitt kernels for edge detection
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = kernel_x.T
        
    elif param == 'laplacian':
        # Laplacian kernel for edge detection
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
        return myConv2(A, kernel, 'same')
    
    # Compute gradients for Sobel and Prewitt
    grad_x = myConv2(A, kernel_x, 'same')
    grad_y = myConv2(A, kernel_y, 'same')
    
    # Combine gradients to get the edge magnitude
    return np.sqrt(grad_x**2 + grad_y**2)

def main(image_path, output_dir='output'):
    # Create a directory for the results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'results_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read the image: {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image to the range [0,1]
    img = img.astype(float) / 255.0
    
    # Convert to grayscale
    A = myColorToGray(img)
    
    # Define combinations of parameters
    noise_types = ['gaussian', 'saltandpepper']
    filters = ['mean', 'median']
    edge_methods = ['sobel', 'prewitt', 'laplacian']
    
    # Loop through all combinations
    for noise_type in noise_types:
        # Add noise
        noisy_img = myImNoise(A, noise_type, std=0.1 if noise_type == 'gaussian' else 0, amount=0.05)
        save_image(noisy_img, f'noisy_{noise_type}', output_dir)
        
        for filter_type in filters:
            # Apply filtering
            filtered_img = myImFilter(noisy_img, filter_type, kernel_size=3)
            save_image(filtered_img, f'filtered_{noise_type}_{filter_type}', output_dir)
            
            for edge_method in edge_methods:
                # Perform edge detection
                edges = myEdgeDetection(filtered_img, edge_method)
                save_image(edges, f'edges_{noise_type}_{filter_type}_{edge_method}', output_dir)
    
    print(f"\nResults saved in the folder: {output_dir}")

def save_image(img, name, output_dir):
    """Helper function to save images."""
    plt.figure(figsize=(8, 8))
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(name)
    plt.savefig(os.path.join(output_dir, f'{name}.png'), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    try:
        image_path = "1710930130543.jpg"
        main(image_path)
    except Exception as e:
        print(f"Error during processing: {str(e)}")