import unittest
from hamcrest import *

import hashlib
import numpy as np

from notmnist_udacity import notmnist_download, normalize_grayscale


class TestNotMnist(unittest.TestCase):

    def test_download(self):
        notmnist_download()
        assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
          'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
        assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
          'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

    def test_normalize_grayscale(self):

        np.testing.assert_array_almost_equal(
        normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
        [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
        0.125098039216, 0.128235294118, 0.13137254902, 0.9],
        decimal=3)
        
        np.testing.assert_array_almost_equal(
        normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
        [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
        0.896862745098, 0.9])
        
