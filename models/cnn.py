import torch
from torch import nn
from torchvision.models import efficientnet_b0, resnet18, resnet34


class Model(nn.Module):
    def __init__(self, C, name: str, feature_extractor_model: nn.Module) -> None:
        super().__init__()

        if name == "effnet":
            self.in_features = feature_extractor_model.classifier[1].in_features
        else:
            self.in_features = feature_extractor_model.fc.in_features

        self.feature_extractor = nn.Sequential(
            *list(feature_extractor_model.children())[:-1]
        )

        self.batch_norm = nn.BatchNorm2d(self.in_features)

        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=C.LSTM_HIDDEN_DIM,
            num_layers=C.LSTM_NUM_LAYERS,
            dropout=C.LSTM_DROPOUT if C.LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm_out_dim = C.LSTM_HIDDEN_DIM * 2

        self.layer_norm = nn.LayerNorm(self.lstm_out_dim)

        self.attention_pool = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(C.FC_DROPOUT)


        self.linear = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim // 2),
            nn.BatchNorm1d(self.lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(C.FC_DROPOUT),
            nn.Linear(self.lstm_out_dim // 2, C.NUM_CLASSES),
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
        features = self.layer_norm(features)
        features, _ = self.lstm(features)


        attention_weights = self.attention_pool(features)
        features = torch.sum(features * attention_weights, dim=1)
        # features = torch.mean(features, dim=1)  # pool across time

        features = self.dropout(features)
        output = self.linear(features)
        return output


def resnet18_lstm_model(C) -> Model:
    resnet = resnet18(weights="DEFAULT")
    for param in resnet.parameters():
        param.requires_grad = False
    # fine tune the lower layer
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.layer3.parameters():
        param.requires_grad = True
    return Model(C, name="resnet", feature_extractor_model=resnet)


def resnet34_lstm_model(C) -> Model:
    resnet = resnet34(weights="DEFAULT")
    for param in resnet.parameters():
        param.requires_grad = False
    # fine tune the lower layer
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.layer3.parameters():
        param.requires_grad = True
    return Model(C, name="resnet", feature_extractor_model=resnet)


def effnet_b0_model(C) -> Model:
    effnet = efficientnet_b0(weights="DEFAULT")
    for param in effnet.parameters():
        param.requires_grad = False
    for idx in range(6, 8):
        for param in effnet.features[idx].parameters():
            param.requires_grad = True
    return Model(C, name="effnet", feature_extractor_model=effnet)
