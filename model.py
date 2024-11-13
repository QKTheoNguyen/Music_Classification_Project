import torch
from torch import nn
from torchsummary import summary

class CNN_Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        # Dense layer
        self.fc1 = nn.Linear(
            in_features=128*4*58,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

        self.fc = nn.Linear(
            in_features=128*4*58,
            out_features=10
        )

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc(x)
        x = self.softmax(x)
        return x
    
class MusicRecNet(nn.Module):
    def __init__(self, 
                 n_mels, 
                 n_frames, 
                 filters: list):
        super(MusicRecNet, self).__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.filters = filters

        modules = nn.Sequential()

        in_channels = 1

        for i, out_channels in enumerate(filters):
            modules.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
            modules.add_module(f'relu{i}', nn.ReLU())
            modules.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            # modules.add_module(f'dropout{i}', nn.Dropout(0.25))
            modules.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.cnn = modules
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_channels * (n_mels // 2 ** len(filters)) * (n_frames // 2 ** len(filters)), 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.dense_2 = nn.Linear(10, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.softmax(x)
        return x


    
if __name__ == "__main__":
    
    # model = CNN_Network()
    # summary(model, (1, 64, 938))

    # print(f'----- CNN_Network model -----')
    # x = torch.randn(1, 64, 938)
    # y = model(x.unsqueeze(1))
    # print(f'Input : {x.size()}')
    # print(f'Output : {y.size()}')
    # print(f'Output logits : {y}')

    n_mels = 16
    n_samples = 6 * 20050
    n_frames = 1 + (n_samples - 2048) // 512
    print(f'n_frames : {n_frames}, n_samples : {n_samples}, n_mels : {n_mels}')
    model_RecNet = MusicRecNet(n_mels=n_mels, n_frames=n_frames, filters=[64, 32, 32, 16])
    summary(model_RecNet, (1, n_mels, n_frames))

    print(f'----- MusicRecNet model -----')
    x = torch.randn(32, 1, n_mels, n_frames)
    y = model_RecNet(x)
    print(f'Input : {x.size()}')
    print(f'Output : {y.size()}')

     