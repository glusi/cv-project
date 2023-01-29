
import zipfile
import shutil
from tqdm import tqdm
import requests
import os


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)

    save_response_content(response, destination)


def save_response_content(response, destination):
    CHUNK_SIZE = 1024
    with open(destination, 'wb') as f:
        total_length = int(response.headers.get('content-length'))
        with tqdm(total=total_length, unit_scale=True, unit='B', unit_divisor=1024) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    pbar.update(CHUNK_SIZE)
                    f.write(chunk)
                    f.flush()


def dowload_h5py_db():
    DB_FILE = 'SynthText_train.h5'
    if not os.path.isfile(DB_FILE):
        ZIP_FILE = 'data.zip'
        file_id = '18vF7hraVDgyK8vFM0qViCpCmM_ja9Bxl'
        destination = ZIP_FILE
        if not os.path.isfile(ZIP_FILE):
            print('Downloading DB file...')
            download_file_from_google_drive(file_id, destination)

        print('Extracting zip file to filesystem...')
        archive = zipfile.ZipFile('data.zip')
        for file in tqdm(archive.namelist()):
            if file.startswith('Project/'):
                archive.extract(file, '.')

        print("Cleaning up...")
        shutil.move('Project/images', './images')
        shutil.move('Project/' + DB_FILE, './' + DB_FILE)
        os.remove(ZIP_FILE)
        shutil.rmtree('Project/')
