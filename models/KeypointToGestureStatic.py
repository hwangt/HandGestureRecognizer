import torch.nn as nn
import torch.nn.functional as F
import torch

class KeypointToGestureStatic(nn.Module):
    def __init__(self):
        super(KeypointToGestureStatic, self).__init__()
        self.fc1 = nn.Linear(21*3, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 5)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def load_checkpoint_for_inference(self, file):
        #Assumes loading from file with dict containing various kvp including the model_state_dict
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()

    def load_model_for_inference(self, file):
        #Assumes loading from file containing model.state_dict()
        model = torch.load(file)
        self.load_state_dict(model)
        self.eval()
