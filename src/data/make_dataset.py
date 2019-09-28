import requests
import zipfile
import os
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32*1024
    total_size = int(response.headers.get('content-length', 0))

    with tqdm(desc=destination, total=total_size, unit='B', unit_scale=True) as pbar:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    pbar.update(CHUNK_SIZE)
                    f.write(chunk)


def download_word2vec(download_dir, gg_drive_id):
    # Download pre-trained word2vec embeddings from google drive
    print("Start downloading pre-trained word2vec embeddings.")
    download_file_name = "ja-gensim_update.txt.zip"

    # file_id = "1ViflLHKz_sQEioELGp7xromuXJsPJd4Y"
    destination = os.path.join(download_dir, download_file_name)
    download_file_from_google_drive(gg_drive_id, destination)
    print("Finish downloading pre-trained word2vec embeddings.")

    # Extract zip file
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(download_dir)
    zip_ref.close()

    print("Delete .zip file.")
    os.remove(destination)


def download_raw_data(destination, gg_drive_id):
    # Download pre-trained word2vec embeddings from google drive
    print("Start downloading raw dataset.")

    # file_id = "1ViflLHKz_sQEioELGp7xromuXJsPJd4Y"
    download_file_from_google_drive(gg_drive_id, destination)
    print("Finish downloading raw dataset from operators.")