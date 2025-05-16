import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# from collections import Counter

from model_object import (UNET, freeze_unet, load_pretrained_weights_1)
from yoloLoss import YoloLoss
from utils import (save_checkpoint, get_bboxes, mean_average_precision)
from dataset import ObjDataset



# Transformação personalizada para VOCDataset
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image = t(image)
        return image, boxes



# Função de treinamento adaptada
def train_fn(train_loader, model, optimizer, loss_fn, device):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            detection_output, _ = model(x)  # Novo forward retorna tupla
            loss = loss_fn(detection_output, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    return sum(mean_loss) / len(mean_loss)

# Configuração principal
def main():
    # Hiperparâmetros
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0
    EPOCHS = 50
    NUM_WORKERS = 4
    PIN_MEMORY = True
    LOAD_MODEL = False
    LOAD_MODEL_FILE = "unet_yolo.pth.tar"
    IMG_DIR = "data/images"
    LABEL_DIR = "data/labels"
    PRETRAINED_PATH = "path_to_pretrained_weights.pth"  # Ajuste para o caminho real

    # Transformações
    transform = Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carregar modelo
    model = UNET(in_channels=3, out_channels=1, split_size=7, num_boxes=2, num_classes=20)
    load_pretrained_weights_1(model, PRETRAINED_PATH)
    freeze_unet(model)
    model = model.to(DEVICE)

    # Verificar parâmetros treináveis
    print("Parâmetros treináveis:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss().to(DEVICE)

    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_MODEL_FILE)
        model.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])

    # Datasets e Dataloaders
    train_dataset = ObjDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    test_dataset = ObjDataset(
        "data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=True,
    )

    # Treinamento
    for epoch in range(EPOCHS):
        # Treinar
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Mean YOLO Loss: {mean_loss:.4f}")

        # Calcular mAP
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Test mAP: {mean_avg_prec:.4f}")

        # Salvar checkpoint
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }
        # save_checkpoint(checkpoint, filename=f"unet_yolo_epoch_{epoch}.pth.tar")
        
        if mean_avg_prec > 0.9:
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
           save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
           import time
           time.sleep(10)
        

    print("Training completed.")

if __name__ == "__main__":
    main()