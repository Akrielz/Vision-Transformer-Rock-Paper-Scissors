import wget
import os
from zipfile import ZipFile


__TRAIN_URL__ = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
__TEST_URL__ = "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip"

__FILE_URLS__ = [__TRAIN_URL__, __TEST_URL__]

__STORAGE_RAW_PATH__ = os.path.normpath("./data_manager/storage/raw")
__STORAGE_ZIP_PATH__ = os.path.normpath("./data_manager/storage/zip")

__TRAIN_FILE_NAME__ = "train.zip"
__TEST_FILE_NAME__ = "test.zip"

__FILE_NAMES__ = [__TRAIN_FILE_NAME__, __TEST_FILE_NAME__]

__IMAGE_SIZE__ = 300
__IMAGE_SHAPE__ = (__IMAGE_SIZE__, __IMAGE_SIZE__, 3)


def download_files():
    if not os.path.isdir(__STORAGE_ZIP_PATH__):
        os.makedirs(__STORAGE_ZIP_PATH__)

    for file_name, file_url in zip(__FILE_NAMES__, __FILE_URLS__):
        full_file_path = os.path.join(__STORAGE_ZIP_PATH__, file_name)

        if not os.path.isfile(full_file_path):
            wget.download(__TRAIN_URL__, full_file_path)


def extract_data():
    if os.path.isdir(__STORAGE_RAW_PATH__):
        return

    for file_name in __FILE_NAMES__:
        full_zip_path = os.path.join(__STORAGE_ZIP_PATH__, file_name)

        path_to_extract_to = os.path.join(__STORAGE_RAW_PATH__, file_name[:file_name.find(".")])

        with ZipFile(full_zip_path, 'r') as fd:
            fd.extractall(path_to_extract_to)


def get_data_local():
    download_files()
    extract_data()
