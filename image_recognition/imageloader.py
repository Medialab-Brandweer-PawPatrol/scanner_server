import tensorflow as tf
import numpy as np
from PIL import Image

# image and mask paths
image_path = 'data/images/WaxO18.JPG'
mask_path = 'data/masks/Wax18.png'

# assign values to pixel colors
def get_mask_layer(px):
    [r,g,b] = px
    if (r > g) and (r > b):  # red
        return 1
    elif (g > b):  # green
        return 2
    else:
        return 0  # other colors default to 0

# load images convert and resize them
def load_image(image_path, mask_path):
    # image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((300, 432))
    image = np.array(image)
    
    # mask
    mask = Image.open(mask_path).convert('RGB')
    mask = mask.resize((300, 432))
    mask = np.array(mask)

    # convert mask colors to values
    mask = np.apply_along_axis(lambda px: get_mask_layer(px), 2, mask)

    return image, mask

# normalize image
def normalize(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)
    return image, mask

# load and normalize the images
image, mask = load_image(image_path, mask_path)
image, mask = normalize(image, mask)

# convert the TensorFlow tensors back to numpy arrays for displaying
image_np = image.numpy()
mask_np = mask.numpy()

# convert the numpy arrays to PIL images
image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

# convert numerical mask values back to an RGB image for visualization/testing
mask_colors = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
mask_colors[mask_np == 1] = [160, 32, 240]    # red to purple
mask_colors[mask_np == 2] = [225, 165, 0]  # green to orange
mask_colors[mask_np == 0] = [0, 225, 0]    # blue to green

mask_pil = Image.fromarray(mask_colors)

# display the images using Pillow
image_pil.show(title='Input Image')
mask_pil.show(title='Mask')
