import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], split_size=7, num_boxes=2, num_classes=20):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Parâmetros do YOLO
        self.S = split_size  # Tamanho da grade (S × S)
        self.B = num_boxes   # Número de bounding boxes por célula
        self.C = num_classes # Número de classes
        
        # Down part of UNET
        for feature in features:
            dropout_rate = 0.1 if feature <= 128 else 0.2
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature
            
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            dropout_rate = 0.1 if feature <= 128 else 0.2
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate))
            
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=0.3)
        
        # Camada final para segmentação
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Detection head híbrido otimizado para Jetson Nano
        self.detection_head = nn.Sequential(
            # Camada 1: Reduzir canais e extrair características
            nn.Conv2d(
                features[-1]*2,  # Canais do bottleneck (512)
                128,             # Menos canais para eficiência
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # Camada 2: Refinar características
            nn.Conv2d(
                128,
                64,              # Reduzir ainda mais
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # Camada 3: Prever C + B*5 (em vez de B*(4+1+C))
            nn.Conv2d(
                64,
                self.C + self.B * 5,  # 20 + 2*5 = 30 (20 classes + 2 caixas com x, y, w, h, confiança)
                kernel_size=1
            )
        )
        
        # Ajuste de resolução para grade S × S
        self.grid_adjust = nn.AdaptiveAvgPool2d((self.S, self.S))  # Força saída S × S
        
        self.initialize_weights()

    def forward(self, x):
        skip_connections = []

        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Detection head: prever bounding boxes, confiança e classes
        detection_output = self.detection_head(x)  # Shape: (batch, C + B*5, H/16, W/16)
        detection_output = self.grid_adjust(detection_output)  # Ajusta para S × S
        detection_output = detection_output.view(-1, self.S, self.S, self.C + self.B * 5)  # Shape: (batch, S, S, 30)

        # Upsampling path (para segmentação)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        segmentation_output = self.final_conv(x)  # Shape: (batch, out_channels, H, W)

        return detection_output, segmentation_output  # Retorna ambas as saídas

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def freeze_unet(model):
    """Congela todas as camadas exceto o detection head e grid_adjust."""
    for name, param in model.named_parameters():
        if "detection_head" not in name and "grid_adjust" not in name:
            param.requires_grad = False


# def load_pretrained_weights(model, pretrained_path):
#     """Carrega pesos pré-treinados, ignorando o detection head."""
#     pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
#     model_dict = model.state_dict()
#     # Filtra pesos incompatíveis (ex.: detection_head)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)



def load_pretrained_weights_1(model, pretrained_path):
    """Carrega pesos pré-treinados, ignorando pesos incompatíveis."""
    # Carrega o ficheiro, que pode ser .pth ou .pth.tar
    data = torch.load(pretrained_path, map_location=torch.device('cpu'))
    
    # Verifica se o ficheiro contém um state_dict diretamente ou um checkpoint
    if isinstance(data, dict) and "state_dict" in data:
        pretrained_dict = data["state_dict"]  # Extrai o state_dict do checkpoint
    else:
        pretrained_dict = data  # Assume que é um state_dict diretamente
    
    model_dict = model.state_dict()
    # Filtra pesos incompatíveis
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)