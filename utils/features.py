import torch
import collections

# https://github.com/metalwhale/hand_tracking/blob/b2a650d61b4ab917a2367a05b85765b81c0564f2/run.py
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (5, 6), (6, 7), (7, 8),
#     (9, 10), (10, 11), (11, 12),
#     (13, 14), (14, 15), (15, 16),
#     (17, 18), (18, 19), (19, 20),
#     (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
# ]

KEYPOINT_SEGMENTS = collections.OrderedDict()
KEYPOINT_SEGMENTS["0_1"] = (0, 1)
KEYPOINT_SEGMENTS["1_2"] = (1, 2)
KEYPOINT_SEGMENTS["2_3"] = (2, 3)
KEYPOINT_SEGMENTS["3_4"] = (3, 4)
KEYPOINT_SEGMENTS["0_5"] = (0, 5)
KEYPOINT_SEGMENTS["5_6"] = (5, 6)
KEYPOINT_SEGMENTS["6_7"] = (6, 7)
KEYPOINT_SEGMENTS["7_8"] = (7, 8)
KEYPOINT_SEGMENTS["0_9"] = (0, 9)
KEYPOINT_SEGMENTS["9_10"] = (9, 10)
KEYPOINT_SEGMENTS["10_11"] = (10, 11)
KEYPOINT_SEGMENTS["11_12"] = (11, 12)
KEYPOINT_SEGMENTS["0_13"] = (0, 13)
KEYPOINT_SEGMENTS["13_14"] = (13, 14)
KEYPOINT_SEGMENTS["14_15"] = (14, 15)
KEYPOINT_SEGMENTS["15_16"] = (15, 16)
KEYPOINT_SEGMENTS["0_17"] = (0, 17)
KEYPOINT_SEGMENTS["17_18"] = (17, 18)
KEYPOINT_SEGMENTS["18_19"] = (18, 19)
KEYPOINT_SEGMENTS["19_20"] = (19, 20)
KEYPOINT_SEGMENTS["0_2"] = (0, 2)
KEYPOINT_SEGMENTS["0_17"] = (0, 17)
KEYPOINT_SEGMENTS["5_17"] = (5, 17)

SEGMENT_ANGLES = [
    ("0_1", "1_2"),  # thumb 1
    ("1_2", "2_3"),  # thumb 2
    ("2_3", "3_4"),  # thumb 3
    ("0_5", "5_6"),  # pointer 1
    ("5_6", "6_7"),  # pointer 2
    ("6_7", "7_8"),  # pointer 3
    ("0_9", "9_10"),  # middle 1
    ("9_10", "10_11"),  # middle 2
    ("10_11", "11_12"),  # middle 3
    ("0_13", "13_14"),  # ring 1
    ("13_14", "14_15"),  # ring 2
    ("14_15", "15_16"),  # ring 3
    ("0_17", "17_18"),  # pinky 1
    ("17_18", "18_19"),  # pinky 2
    ("18_19", "19_20"),  # pinky 3
    ("0_2", "0_17"),  # palm fold
]


def get_seg(keypoints_3d, segments, segment_name):
    return torch.sub(keypoints_3d[segments[segment_name][1]], keypoints_3d[segments[segment_name][0]])


def angle_features(keypoints_3d, segments=KEYPOINT_SEGMENTS, angles=SEGMENT_ANGLES):
    '''
    Given
    :param keypoints_3d: tensor with 21 3d keypoints
    :param connections: list of 20 pairs/tuples of keypoint indices defining hand or finger segments
    :return: tensor of 15 inter segment angles
    '''

    features = []

    # If this is a (1, 21, 3) tensor, squeeze dim 0 because we need to operate on (21,3)
    if len(keypoints_3d.shape) == 3:
        keypoints_3d_shaped = torch.squeeze(keypoints_3d, 0)
    else:
        keypoints_3d_shaped = keypoints_3d
    for seg_first, seg_second in angles:
        vec_first = get_seg(keypoints_3d_shaped, segments, seg_first)
        vec_second = get_seg(keypoints_3d_shaped, segments, seg_second)
        t = torch.dot(vec_first, vec_second) / (torch.norm(vec_first) * torch.norm(vec_second))
        t= torch.clamp(t, min=-1.0, max=1.0) #need to clamp to valid range for numerical stability
        angle = torch.acos(t)
        features.append(angle)
    # return torch.rad2deg(torch.stack(features, dim=0))
    return torch.stack(features, dim=0)


def palm_normal(keypoints_3d, segments=KEYPOINT_SEGMENTS):
    '''
    Compute palm normal direction as defined by cross product of segment (0, 5) X (0, 17)
    :param keypoints_3d:
    :return:
    '''
    # If this is a (1, 21, 3) tensor, squeeze dim 0 because we need to operate on (21,3)
    if len(keypoints_3d.shape) == 3:
        keypoints_3d_shaped = torch.squeeze(keypoints_3d, 0)
    else:
        keypoints_3d_shaped = keypoints_3d

    return torch.cross(get_seg(keypoints_3d_shaped, segments, "0_5"), get_seg(keypoints_3d_shaped, segments, "0_17"))
