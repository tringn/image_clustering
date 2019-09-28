import os
from src.utils.config import get_config_from_json
from src.models.feature_extractor import run_inference_on_images_feature

IMAGE_EXTENSION = ('jpg', 'jpeg', 'bmp', 'png')


def extract_feature(img_dir, model_dir, output_dir):
    """
    Extract image features of all images in img_dir and save feature vectors to output_dir
    :param img_dir: (string) directory containing images to extract feature
    :param model_dir: (string) directory containing extractor model
    :param output_dir: (string) directory to save feature vector file
    :return:
    """
    # Get list of image paths
    img_list = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir) if img_file.endswith(IMAGE_EXTENSION)]

    # Run getting feature vectors for each image
    run_inference_on_images_feature(img_list, model_dir, output_dir)


if __name__ == "__main__":
    # Get config
    config, _ = get_config_from_json("configs/configs.json")

    object_name = config.model.object_name
    img_dir = os.path.join(config.paths.image_dir, object_name)
    vector_dir = os.path.join(config.paths.vector_dir, object_name)

    # Extract feature for all images in image directory
    extract_feature(img_dir, config.paths.model_dir, vector_dir)
