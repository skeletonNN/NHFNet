import torch.nn as nn

'''
Input output shapes
visual (FACET): (20, 35)
label: (1) -> [sentiment]
'''


class FACETVisualLSTMNet(nn.Module):
    def __init__(self, features_only=False):
        super(FACETVisualLSTMNet, self).__init__()
        self.features_only = features_only
        self.lstm1 = nn.LSTM(input_size=35, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        last = x[:, -1, :]

        return self.classifier(last)
