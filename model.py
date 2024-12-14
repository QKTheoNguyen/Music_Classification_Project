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
                 filters: list,
                 add_dropout: bool = False):
        super(MusicRecNet, self).__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.filters = filters
        self.add_dropout = add_dropout

        modules = nn.Sequential()

        in_channels = 1

        for i, out_channels in enumerate(filters):
            modules.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
            modules.add_module(f'relu{i}', nn.ReLU())
            modules.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            if add_dropout:
                modules.add_module(f'dropout{i}', nn.Dropout(0.25))
            modules.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.cnn = modules
        self.flatten = nn.Flatten()
        dense_out = 16
        self.dense = nn.Linear(in_channels * (n_mels // 2 ** len(filters)) * (n_frames // 2 ** len(filters)), dense_out)
        self.relu = nn.ReLU()
        if add_dropout:
            self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(dense_out)
        self.dense_2 = nn.Linear(dense_out, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        if self.add_dropout:
            x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.dense_2(x)
        x = self.softmax(x)
        return x

class CNN_new(nn.Module):
    def __init__(self,
                 n_mels,
                 n_frames):
        super(CNN_new, self).__init__()

        modules = nn.Sequential()

        modules.add_module('conv1', nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)))
        modules.add_module('relu1', nn.ReLU())
        modules.add_module('maxpool1', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)))
        modules.add_module('batchnorm1', nn.BatchNorm2d(64))

        modules.add_module('conv2', nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)))
        modules.add_module('relu2', nn.ReLU())
        modules.add_module('maxpool2', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)))
        modules.add_module('batchnorm2', nn.BatchNorm2d(32))

        modules.add_module('conv3', nn.Conv2d(32, 32, kernel_size=(2, 2), padding=(0, 0)))
        modules.add_module('relu3', nn.ReLU())
        modules.add_module('maxpool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        modules.add_module('batchnorm3', nn.BatchNorm2d(32))

        modules.add_module('conv4', nn.Conv2d(32, 16, kernel_size=(1, 1), padding=(0, 0)))
        modules.add_module('relu4', nn.ReLU())
        modules.add_module('maxpool4', nn.MaxPool2d(kernel_size=(1, 1), stride=(2, 2)))
        modules.add_module('batchnorm4', nn.BatchNorm2d(16))

        modules.add_module('flatten', nn.Flatten())
        modules.add_module('dense1', nn.Linear(16 * (n_mels // 2 ** 4) * (n_frames // 2 ** 4), 64))
        modules.add_module('relu5', nn.ReLU())
        # modules.add_module('batchnorm5', nn.BatchNorm1d(64))
        modules.add_module('dropout', nn.Dropout(0.3))
        modules.add_module('dense2', nn.Linear(64, 10))
        modules.add_module('softmax', nn.Softmax(dim=1))

        self.cnn = modules

    def forward(self, x):
        return self.cnn(x)



if __name__ == "__main__":
    
    # model = CNN_Network()
    # summary(model, (1, 64, 938))

    # print(f'----- CNN_Network model -----')
    # x = torch.randn(1, 64, 938)
    # y = model(x.unsqueeze(1))
    # print(f'Input : {x.size()}')
    # print(f'Output : {y.size()}')
    # print(f'Output logits : {y}')

    # n_mels = 16
    # n_samples = 6 * 20050
    # n_frames = 1 + (n_samples - 2048) // 512
    # print(f'n_frames : {n_frames}, n_samples : {n_samples}, n_mels : {n_mels}')
    # model_RecNet = MusicRecNet(n_mels=n_mels, n_frames=n_frames, filters=[32, 64, 128])
    # summary(model_RecNet, (1, n_mels, n_frames))

    # print(f'----- MusicRecNet model -----')
    # x = torch.randn(32, 1, n_mels, n_frames)
    # y = model_RecNet(x)
    # print(f'Input : {x.size()}')
    # print(f'Output : {y.size()}')

    n_mels = 16
    n_samples = 6 * 20050
    n_frames = 1 + (n_samples - 2048) // 512
    print(f'n_frames : {n_frames}, n_samples : {n_samples}, n_mels : {n_mels}')
    model_CNN_new = CNN_new(n_mels=n_mels, n_frames=n_frames)
    summary(model_CNN_new, (1, n_mels, n_frames))

    print(f'----- CNN_new model -----')
    x = torch.randn(32, 1, n_mels, n_frames)
    y = model_CNN_new(x)
    print(f'Input : {x.size()}')
    print(f'Output : {y.size()}')

     