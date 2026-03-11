import numpy as np
import matplotlib.pyplot as plt
import os

##### Data Loading #####

def load_reconstruction(path):
    """Loads a reconstructed HSI cube from a .npz file."""
    with np.load(path) as data:
        return data['data']
    
##### Image Rendering #####

def render_band(hsi, band_idx):
    """
    Converts a raw HSI band into a normalized 2D image array.
    
    Returns
    -------
    numpy.ndarray
        The 2D image slice.
    """
    if band_idx < 0 or band_idx >= hsi.shape[2]:
        raise IndexError(f"Band index {band_idx} out of range.")
    
    slice_2d = hsi[:, :, band_idx]
    return slice_2d

##### Display Logic #####

def show_images(images, titles=None, cmap='gray'):
    """
    Displays one or more images side-by-side.
    
    Parameters
    ----------
    images : list of numpy.ndarray or single numpy.ndarray
        The image(s) to display.
    titles : list of str or str, optional
        Titles for each image.
    cmap : str
        Colormap (default 'gray').
    """
    if not isinstance(images, list):
        images = [images]
    if titles and not isinstance(titles, list):
        titles = [titles]

    num_imgs = len(images)
    fig, axes = plt.subplots(1, num_imgs, figsize=(6 * num_imgs, 5), squeeze=False)

    for i in range(num_imgs):
        ax = axes[0, i]
        img = ax.imshow(images[i], cmap=cmap)
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
            

    plt.tight_layout()
    plt.show()


##### Composite Helper #####

def compare_reconstructions_by_paths(paths, band_idx, titles=None):
    """
    Convenience function: pass a list of .npz paths and see them side-by-side.
    """
    rendered_list = []
    final_titles = titles if titles else [os.path.basename(p) for p in paths]
    
    for p in paths:
        cube = load_reconstruction(p)
        rendered_list.append(render_band(cube, band_idx))
    
    show_images(rendered_list, titles=final_titles)


##### Data Manipulation ######

def to_false_color(hsi: np.ndarray, bands: list = None) -> np.ndarray:
    """
    Converts a Hyperspectral cube into an RGB false-color image.
    
    Args:
        hsi: Input cube of shape (H, W, C).
        bands: Optional list of 3 indices [R, G, B]. 
               Defaults to 30%, 50%, and 70% of the spectral range.
               
    Returns:
        Normalized RGB image of shape (H, W, 3) in range [0, 1].
    """
    num_bands = hsi.shape[2]
    
    # 1. Default band selection (evenly spaced through the cube)
    if bands is None:
        bands = [int(num_bands * 0.3), int(num_bands * 0.5), int(num_bands * 0.7)]
    
    if len(bands) != 3:
        raise ValueError("The 'bands' argument must contain exactly 3 indices.")

    # 2. Extract and handle potential complex values (e.g., from FFT)
    rgb = hsi[:, :, bands].real.astype(np.float32)
    
    # 3. Per-band Normalization
    # We normalize each channel individually to [0, 1] to ensure 
    # the image "pops" regardless of the original data's bit-depth.
    for i in range(3):
        channel = rgb[:, :, i]
        c_min, c_max = channel.min(), channel.max()
        
        # Avoid division by zero for empty/flat bands
        denom = (c_max - c_min) + 1e-8
        rgb[:, :, i] = (channel - c_min) / denom
        
    return rgb