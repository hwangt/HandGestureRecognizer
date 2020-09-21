from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from utils.features import angle_features, palm_normal, KEYPOINT_SEGMENTS, SEGMENT_ANGLES


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class HdgDataset(Dataset):
    """
    HandDatasetGenerator dataset
    """
    def __init__(self,
                 root_dir='/tmp',
                 transforms=None):
        """
        Args:
            root_dir (string): Root directory. Expected structure: <root_dir>/<gesture>/YYYYMMDD-HHMMSS_<user>/
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Collect the gesture labels, which are the top level directories
        self.gesture_labels = next(os.walk(root_dir))[1]
        self.gesture_labels = sorted(self.gesture_labels)

        # Encode gesture_labels as numeric
        label_encoder = preprocessing.LabelEncoder()
        self.targets = label_encoder.fit_transform(self.gesture_labels)

        self.root_dir = root_dir
        self.keypoints_files = []
        self.labels = []
        self._get_keypoints_and_labels()

        self.transforms = transforms

    def _get_keypoints_and_labels(self):
        '''
        Loads keypoint filenames and labels from stored tensors found in each directory of the form:
        <root_dir>/<gesture>/YYYYMMDD-HHMMSS_<user>/
        e.g.
        keypoint = "/tmp/thumbs_up/20200920-181510_tony/###_norm_landmarks.pt"
        label = "thumbs_up"

        :return:
        keypoint_list
        label_list
        '''

        #Extensions to load
        extensions = tuple(["_norm_landmarks.pt"])

        for idx in range(len(self.gesture_labels)):
            keypoints_dir = os.path.join(self.root_dir, self.gesture_labels[idx])
            keypoints_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(keypoints_dir) for f in filenames if f.endswith(extensions)]
            labels = [self.targets[idx]] * len(keypoints_files)
            self.keypoints_files += keypoints_files
            self.labels += labels
            print(f'{self.gesture_labels[idx]}')
            print(f'num samples {len(keypoints_files)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = torch.load(self.keypoints_files[idx])
        angles = angle_features(X, segments=KEYPOINT_SEGMENTS, angles=SEGMENT_ANGLES)  # intersegment angles
        normal = palm_normal(X, segments=KEYPOINT_SEGMENTS)  # palm normal vector

        if len(X.shape) == 3: #if keypoints are stored as (1,21,3) vs (21,3)
            keypt_volume = torch.reshape(X[0][1:], (5,4,3)) #exclude base of palm, stack finger keypoints
        else:
            keypt_volume = torch.reshape(X[1:], (5,4,3))

        y = torch.as_tensor(self.labels[idx])
        return keypt_volume, angles, normal, y
