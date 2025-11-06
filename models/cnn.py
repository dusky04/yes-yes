import torch
from torch import nn
from torchvision.models import efficientnet_b0, resnet18


class Model(nn.Module):
    def __init__(self, C, name: str, feature_extractor_model: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            *list(feature_extractor_model.children())[:-1]
        )
        if name == "effnet":
            self.in_features = feature_extractor_model.classifier[1].in_features
        else:
            self.in_features = feature_extractor_model.fc.in_features

        self.batch_norm = nn.BatchNorm2d(num_features=512)
        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=C.LSTM_HIDDEN_DIM,
            num_layers=C.LSTM_NUM_LAYERS,
            dropout=C.LSTM_DROPOUT if C.LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(C.FC_DROPOUT)
        lstm_out_dim = C.LSTM_HIDDEN_DIM * 2

        self.linear = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.BatchNorm1d(lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(C.FC_DROPOUT),
            nn.Linear(lstm_out_dim // 2, C.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input dim: [batch_size, frame, C, H, W]
        B, T, C, H, W = x.shape

        # since we are passing through conv2D layers in ResNet
        # have to convert dims to [batch_size * frame, C, H, W]
        # output dims: [batch_size, 512 (in_features), 1, 1]
        features = self.feature_extractor(x.view(B * T, C, H, W))
        features = self.batch_norm(features)

        # LSTM expects (batch_size, sequence_length, input_size) as input
        features = features.view(B, T, -1)  # [batch_size, sequence_length, 512]
        # output of LSTM dims: [batch_size, sequence_length, hidden_dim]
        features, _ = self.lstm(features)
        features = torch.mean(features, dim=1)

        features = self.dropout(features)
        output = self.linear(features)
        return output


def resnet_lstm_model(C) -> Model:
    resnet = resnet18(weights="DEFAULT")
    # fine tune the lower layer
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.layer3.parameters():
        param.requires_grad = True
    return Model(C, name = "resnet", feature_extractor_model=resnet)


def effnet_b0_model(C) -> Model:
    effnet = efficientnet_b0(weights="DEFAULT")
    for idx in range(6, 8):
        for param in effnet.features[idx].parameters():
            param.requires_grad = True
    return Model(C, name = "effnet", feature_extractor_model=effnet)
