import pandas as pd
import os
import json
import shutil
THRESHOLD = 0.85


def get_object_type(object_type_path):
    """
    Read object type (.csv) file and get list of type in object master
    :param label_path: (string) path to object master type
    :return: list of types
    """
    df = pd.read_csv(object_type_path)

    types = df[["type", "object_type"]].drop_duplicates().reset_index().sort_values(by=['type', 'object_type'])
    print(types)
    types[["type", "object_type"]].to_csv("../../results/types.csv", index=False)
    return df


def symlink_cluster(label_path, dest_dir, src_dir):
    """
    Link the images in img_root to symlink_dir with its cluster defined in label_path
    :param label_path: (string) path to json file containing cluster label of image
    :param dest_dir: (string) destination directory to link image
    :param src_dir: (string) source directory to link image
    :return:
    """
    with open(label_path, "r") as f:
        json_dat = json.load(f)

    df = pd.DataFrame(json_dat['data'])

    # Convert prob string to float
    df["prob"] = df["prob"].apply(lambda x: float(x))

    top_labels = df[df["prob"] >= THRESHOLD]
    top_labels_count = top_labels["label"].value_counts()

    # Make symlink dir
    os.makedirs(dest_dir, exist_ok=True)

    print("Object %s has %d images" % (label_path, len(df)))
    print(top_labels_count)
    print("\n")

    # Remove previous symlink
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    for l in top_labels_count.index:
        label_name = l.replace("/", "") + "_" + str(top_labels_count[l])

        # Create folder for label
        os.makedirs(os.path.join(dest_dir, label_name), exist_ok=True)

        img_files = top_labels[top_labels["label"] == l]["img_file"].values.tolist()

        for img_file in img_files:
            src_img_path = os.path.abspath(os.path.join(src_dir, img_file))
            dst_img_path = os.path.join(dest_dir, label_name, img_file)
            os.symlink(src_img_path, dst_img_path)


def symlink_objects(img_json_dir, dest_root_dir, src_root_dir):
    """
    Link the images for each object stored in img_json_dir from src_root_dir to dest_root_dir with corresponding cluster labels
    :param img_json_dir: (string) directory of label json files
    :param dest_root_dir: (string) directory of destination to link objects' images
    :param src_root_dir: (string) directory of source to link objects' images
    :return:
    """
    for json_file in os.listdir(img_json_dir):
        if json_file.endswith(".json"):
            object_name = json_file.replace(".json", "")
            label_path = os.path.join(img_json_dir, object_name)
            dest_dir = os.path.join(dest_root_dir, object_name)
            src_dir = os.path.join(src_root_dir, object_name)
            symlink_cluster(label_path, dest_dir, src_dir)


if __name__ == "__main__":
    # Get object type from object master
    # object_type_path = "../../data/interim/object-list-with-type.csv"
    # get_object_type(object_type_path)

    # Symlink top label
    img_json_dir = "../../results/k_means_json"
    symlink_dir = "../../results/k_means"
    img_root = "../../data/raw/images/instagram"
    symlink_objects(img_json_dir, symlink_dir, img_root)
