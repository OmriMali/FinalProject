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