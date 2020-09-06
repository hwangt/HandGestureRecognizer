from __future__ import print_function, division
import os
import cv2
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Include Zak's port of MediaPipe tflite models to PyTorch port
# import sys
# sys.path.append(mediapipe_pytorch_path)

import blazepose
from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS


class ImageToKeypointConverter():
    def __init__(self):
        self.mediapipe_pytorch_path = "/home/thwang/3psrc/MediaPipePyTorch"
        self.palm_weights_path = os.path.join(self.mediapipe_pytorch_path, "blazepalm.pth")
        self.anchors_path=os.path.join(self.mediapipe_pytorch_path, "anchors_palm.npy")
        self.landmarks_weights_path = os.path.join(self.mediapipe_pytorch_path, "blazehand_landmark.pth")
        self.palm_detector = self.load_palm_detector(palm_thresh=0.75)
        self.hand_regressor = self.load_landmark_regressor()

    def load_palm_detector(self, palm_thresh=0.75):
        palm_detector = BlazePalm()
        palm_detector.load_weights(self.palm_weights_path)
        palm_detector.load_anchors(self.anchors_path)
        palm_detector.min_score_thresh = palm_thresh
        return palm_detector

    def load_landmark_regressor(self):
        hand_regressor = BlazeHandLandmark()
        hand_regressor.load_weights(self.landmarks_weights_path)
        return hand_regressor

    def get_landmarks_from_image(self,
                            image_file):
        frame = cv2.imread(image_file)
        img1, img2, scale, pad = resize_pad(frame)

        normalized_palm_detections = self.palm_detector.predict_on_image(img1)
        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

        xc, yc, scale, theta = self.palm_detector.detection2roi(palm_detections)
        img, affine2, box2 = self.hand_regressor.extract_roi(frame, xc, yc, theta, scale)
        flags2, handed2, normalized_landmarks2 = self.hand_regressor(img)
        landmarks2 = self.hand_regressor.denormalize_landmarks(normalized_landmarks2, affine2)

        for i in range(len(flags2)):
            landmark, flag = landmarks2[i], flags2[i]
            print(flag)
            if flag > 0.5:
                draw_landmarks(frame, landmark[:, :2], HAND_CONNECTIONS, size=2)

        return flags2, handed2, normalized_landmarks2

    def write_csv_landmarks_from_images(self, root_dir, subdir_names, image_selection_strategy='ALL' ):
        """
        Given a root directory, for each subdirectory in subdir_names, select images based on strategy,
        and write one CSV file with landmark info the following format:

        TK

        :param root_dir:
        :param subdir_names:
        :param image_selection_strategy: 'all', 
        :return:
        """

        pass

