from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class HgmDataset(Dataset):
    """
    HGM dataset
    """
    def __init__(self,
                 root_dir='/home/thwang/data/HGM-1.0/',
                 transforms=None):
        """
        Args:
            root_dir (string): Root directory containing the four camera folders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # For this dataset, there are four camera views, containing
        # sub-folder names that correspond to gesture labels
        self.camera = "Below_CAM"
        self.gesture_labels = ["F", "G", "L", "O", "Y"]

        # Encode gesture_labels as numeric
        label_encoder = preprocessing.LabelEncoder()
        self.targets = label_encoder.fit_transform(self.gesture_labels)
        print(f'targets{self.targets}')

        self.root_dir = root_dir
        self.keypoints_files = []
        self.labels = []
        self._get_keypoints_and_labels()

        self.transforms = transforms

    def _get_keypoints_and_labels(self):
        '''
        Loads keypoint filenames and labels from stored tensors found in each directory of the form:
        <root_dir>/<camera>/<gesture_label>
        e.g.
        keypoint = "/home/thwang/data/HGM-1.0/Below_CAM/F/P1_001.pt"
        label = "F"

        :return:
        keypoint_list
        label_list
        '''

        #Extensions to load
        extensions = tuple(["pt"])

        for idx in range(len(self.gesture_labels)):
            keypoints_dir = os.path.join(self.root_dir, self.camera, self.gesture_labels[idx])
            keypoints_files = [os.path.join(keypoints_dir, f) for f in sorted(os.listdir(keypoints_dir))
                               if os.path.isfile(os.path.join(keypoints_dir, f))
                               and f.endswith(extensions)]
            labels = [self.targets[idx]] * len(keypoints_files)
            self.keypoints_files += keypoints_files
            self.labels += labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = torch.flatten(torch.load(self.keypoints_files[idx]))
        y = torch.as_tensor(self.labels[idx])

        # if self.transforms:
        # TODO: Consider adding transforms to change feature representation from 21 3d keypoints to relative keypoint
        # offsets, angles, hand orientation, etc.
        #     X = self.transforms(X)

        return X, y