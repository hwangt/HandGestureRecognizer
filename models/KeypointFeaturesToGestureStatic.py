import torch.nn as nn
import torch.nn.functional as F
import torch

class KeypointFeaturesToGestureStatic(nn.Module):
    '''
    Derive features from 21 3D keypoints to predict static gesture.
    Reshape 21 3D keypoints 5C x 4H x 3W (corresponding to 5 fingers x 4 segments/finger x 3D coord )
    and use conv layers, also concat palm normal vector, inter-segment angles.

    '''
    def __init__(self, device='cpu'):
        super(KeypointFeaturesToGestureStatic, self).__init__()

        self.device = device

        keypt_volume_out_channels = 10
        self.conv22 = nn.Conv2d(in_channels=5, out_channels=keypt_volume_out_channels, kernel_size=(2,2), stride=1,)
        nn.init.xavier_uniform(self.conv22.weight)
        self.conv32 = nn.Conv2d(in_channels=5, out_channels=keypt_volume_out_channels, kernel_size=(2,3), stride=1,)
        nn.init.xavier_uniform(self.conv32.weight)
        self.bn22 = nn.BatchNorm2d(num_features=keypt_volume_out_channels,)
        self.bn32 = nn.BatchNorm2d(num_features=keypt_volume_out_channels,)
        self.bn_angles = nn.BatchNorm1d(num_features=16)
        self.bn_normal = nn.BatchNorm1d(num_features=3)
        self.fc1 = nn.Linear(109, 64, )
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32, )
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 7, )


    def forward(self, keypt_volume, angles, normal):
        '''
        :param inputs: Expect inputs to be a list of Tensors of form [keypt_volume, angles, normal]
        :return:
        '''

        out_keypt22 = self.conv22(keypt_volume)
        out_keypt22 = F.relu(self.bn22(out_keypt22))
        out_keypt22 = out_keypt22.reshape(out_keypt22.size(0), -1)

        out_keypt32 = self.conv32(keypt_volume)
        out_keypt32 = F.relu(self.bn32(out_keypt32))
        out_keypt32 = out_keypt32.reshape(out_keypt32.size(0), -1)

        angles = self.bn_angles(angles)

        normal = self.bn_normal(normal)

        merged_feature = torch.cat((out_keypt22, out_keypt32, angles, normal), 1) #109, 1

        out = F.relu(self.bn_fc1(self.fc1(merged_feature))) #64, 1
        out = F.relu(self.bn_fc2(self.fc2(out))) #32, 1
        out = self.fc3(out) #7, 1

        return out

    def load_checkpoint_for_inference(self, file):
        #Assumes loading from file with dict containing various kvp including the model_state_dict
        checkpoint = torch.load(file, map_location=torch.device(self.device))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()

    def load_model_for_inference(self, file):
        #Assumes loading from file containing model.state_dict()
        model = torch.load(file, map_location=torch.device(self.device))
        self.load_state_dict(model)
        self.eval()
