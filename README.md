# Brandweer Scanner Back-end

# Image Preprocessing for Training

This scanner_server backend code is used to process, clean and resize image and mask data for use in TensorFlow's Image Segmentation library. The code handles tasks such as loading images, resizing them, converting masks to numerical values, and normalizing the data. This code is used for preparing data for training Image Recognition models.

## Usage

1. Place your images in the `data/images` directory and your masks in the `data/masks` directory.

2. Run the preprocessing script:
    ```sh
    python imageloader.py
    ```
