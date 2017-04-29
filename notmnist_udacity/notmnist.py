from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle

PICKLE_FILE = 'notMNIST.pickle'

def get_pickle_notmnist_train_valid_test():
    pickle_file = PICKLE_FILE
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        train_features = pickle_data['train_dataset']
        train_labels = pickle_data['train_labels']
        valid_features = pickle_data['valid_dataset']
        valid_labels = pickle_data['valid_labels']
        test_features = pickle_data['test_dataset']
        test_labels = pickle_data['test_labels']
        del pickle_data  # Free up memory
    return train_features, train_labels,\
      valid_features, valid_labels,\
      test_features,test_labels
        
def save_pickle_notmnist_train_valid_test():
    notmnist_download()
    train_features, train_labels, test_features, test_labels \
        = notmnist_uncompress()
    train_features, valid_features, train_labels, valid_labels \
        = train_test_split(
            train_features,
            train_labels,
            test_size=0.05,
            random_state=832289)
    pickle_file = PICKLE_FILE
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('notMNIST.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    
def notmnist_download():
    download(
        'https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip',
        'notMNIST_train.zip')
    download(
        'https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip',
        'notMNIST_test.zip')

def notmnist_uncompress():
    train_features, train_labels\
        = uncompress_features_labels('notMNIST_train.zip')
    test_features, test_labels\
        = uncompress_features_labels('notMNIST_test.zip')
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)

    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    return train_features, train_labels, test_features, test_labels

    
import os
from urllib.request import urlretrieve

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm
import numpy as np

def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    return (image_data/255)*0.8 + 0.1
