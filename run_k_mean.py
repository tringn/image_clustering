import json
import os

from src.models.k_mean import read_vector, reduce_dim_combine, k_mean
from src.utils.analyze_label import symlink_cluster
from src.utils.config import get_config_from_json

IMAGE_EXTENSION = ('jpg', 'jpeg', 'bmp', 'png')

if __name__ == "__main__":

    # Get config
    config, _ = get_config_from_json("configs/configs.json")

    object_name = config.model.object_name

    dim = config.model.reduced_dimension

    img_dir = os.path.join(config.paths.image_dir, object_name)
    vector_dir = os.path.join(config.paths.vector_dir, object_name)
    save_plot_dir = os.path.join(config.paths.plot_dir, object_name)
    cluster_label_path = os.path.join(config.paths.cluster_label_dir, object_name + ".json")

    if not os.path.isdir(vector_dir):
        raise Exception("Please run feature extraction for all images first")

    # Read feature vector from vector dir
    vector_array, vector_files = read_vector(vector_dir)

    if len(vector_files) == 0:
        raise Exception("Please run feature extraction for all images first")

    # Apply dimensional reducing approach
    vector_array = reduce_dim_combine(vector_array, dim=dim)

    labels = k_mean(vector_array, config.model.k).tolist()

    assert len(labels) == len(vector_files), "Not equal length"

    label_dict = [{"img_file": vector_files[i].replace(".npz", ""), "label": str(labels[i]), "prob": "1.0"} for i in
                  range(len(labels))]

    # Save to disk
    os.makedirs(os.path.dirname(cluster_label_path), exist_ok=True)
    with open(cluster_label_path, 'w') as fp:
        json.dump({"data": label_dict}, fp)

    print("Cluster label for each image are saved at results/cluster_label/example.")

    # Symlink
    link_base_dir = config.paths.link_dir
    os.makedirs(link_base_dir, exist_ok=True)

    symlink_cluster(label_path=cluster_label_path,
                    dest_dir=os.path.join(link_base_dir, object_name),
                    src_dir=img_dir)

    print("Go to results/link/example to see images in each cluster")
