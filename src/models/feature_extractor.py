import os
import re
import sys
import tarfile
import zipfile
import numpy as np
import tensorflow as tf
from six.moves import urllib
import psutil
from collections import defaultdict

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
IMAGE_EXTENSION = ('jpg', 'jpeg', 'bmp', 'png')


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, model_dir):
        label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph(model_dir):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, model_dir):
    """Runs inference on an image.

    Args:
      image: Image file name.
      model_dir: Directory contains model

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph(model_dir)

    num_top_predictions = 5

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup(model_dir)

        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))


def run_inference_on_images_feature(image_list, model_dir, output_dir):
    """Runs inference on an image list and get features.
    Args:
      image_list: {list} a list of paths to image files
      model_dir: (string) name of the directory where model is
      output_dir: {string} name of the directory where image vectors will be saved
    Returns:
      save image feature into output_dir
    """
    image_to_labels = defaultdict(list)

    create_graph(model_dir)

    os.makedirs(output_dir, exist_ok=True)

    num_top_predictions = 5

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        for image_index, image in enumerate(image_list):
            try:
                print("parsing", image_index, image, "\n")
                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                with tf.gfile.FastGFile(image, 'rb') as f:
                    image_data = f.read()

                    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    feature_set = sess.run(feature_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    feature_vector = np.squeeze(feature_set)
                    outfile_name = os.path.basename(image) + ".npz"
                    out_path = os.path.join(output_dir, outfile_name)
                    np.savetxt(out_path, feature_vector, delimiter=',')

                # close the open file handlers
                proc = psutil.Process()
                open_files = proc.open_files()

                for open_file in open_files:
                    file_handler = getattr(open_file, "fd")
                    os.close(file_handler)
            except:
                print('could not process image index', image_index, 'image', image)

    return image_to_labels


def maybe_download_and_extract(model_dir, data_url):
    """Download and extract model tar file."""
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if data_url.endswith(".tgz"):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    elif data_url.endswith(".zip"):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_directory)
    else:
        raise ValueError


if __name__ == '__main__':
    model_dir = "models"

    maybe_download_and_extract(model_dir)

    # image = os.path.join(model_dir, 'cropped_panda.jpg')
    # run_inference_on_image(image, model_dir)

    image_dir = "../../scene_recognition/vgg365/data/raw/images/instagram/"
    output_dir = "results/image_vectors/"

    # Get image paths
    object_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]

    for object_path in object_paths:
        image_list = [os.path.join(object_path, file_name) for file_name in os.listdir(object_path) if file_name.endswith(IMAGE_EXTENSION)]
        obj_output_dir = os.path.join(output_dir, os.path.basename(object_path))
        run_inference_on_images_feature(image_list=image_list,
                                        model_dir=model_dir,
                                        output_dir=obj_output_dir)
