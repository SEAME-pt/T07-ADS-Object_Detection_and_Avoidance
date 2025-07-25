import os
import cv2
import albumentations as A
import numpy as np
import shutil
from tqdm import tqdm

def read_yolo_label(label_path):
    """Lê um arquivo de anotações YOLO e retorna as bounding boxes."""
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, width, height, int(class_id)])
    return bboxes

def write_yolo_label(label_path, bboxes):
    """Escreve as bounding boxes no formato YOLO em um arquivo."""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_dataset(input_dataset_dir, output_dataset_dir, num_augmentations=3, save_empty_labels=False):
    """
    Aplica augmentações ao dataset YOLO, gerando novas imagens e labels.
    Não salva imagens sem bounding boxes válidas, a menos que save_empty_labels=True.
    Args:
        input_dataset_dir: Caminho do dataset original (ex.: yolov8_last/).
        output_dataset_dir: Caminho onde o dataset aumentado será salvo.
        num_augmentations: Número de imagens aumentadas por imagem original.
        save_empty_labels: Se True, cria arquivos .txt vazios para imagens sem bboxes válidas.
    """
    # Definir augmentações (compatíveis com bounding boxes)
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

    # Criar estrutura de diretórios no dataset de saída
    os.makedirs(os.path.join(output_dataset_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'valid', 'labels'), exist_ok=True)

    # Copiar o arquivo data.yaml
    shutil.copy(os.path.join(input_dataset_dir, 'data.yaml'), output_dataset_dir)

    # Processar imagens e labels para train e valid
    for split in ['train', 'valid']:
        image_dir = os.path.join(input_dataset_dir, split, 'images')
        label_dir = os.path.join(input_dataset_dir, split, 'labels')
        output_image_dir = os.path.join(output_dataset_dir, split, 'images')
        output_label_dir = os.path.join(output_dataset_dir, split, 'labels')

        if not os.path.exists(image_dir):
            print(f"Diretório não encontrado: {image_dir}")
            continue

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in tqdm(image_files, desc=f"Processando {split}"):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erro ao carregar imagem: {image_path}")
                continue

            # Carregar labels, se existirem
            bboxes = read_yolo_label(label_path)
            class_ids = [int(bbox[4]) for bbox in bboxes] if bboxes else []
            bboxes = [bbox[:4] for bbox in bboxes] if bboxes else []

            # Copiar imagem e label originais, se houver labels ou se save_empty_labels=True
            if bboxes or save_empty_labels:
                shutil.copy(image_path, os.path.join(output_image_dir, image_file))
                if bboxes:
                    shutil.copy(label_path, os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt'))
                elif save_empty_labels:
                    # Criar .txt vazio para imagens sem labels
                    open(os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt'), 'a').close()

            # Gerar imagens aumentadas
            for i in range(num_augmentations):
                # Aplicar augmentação
                augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_ids = augmented['class_ids']

                # Combinar bboxes com class_ids para salvar no formato YOLO
                aug_bboxes_with_class = [
                    [bbox[0], bbox[1], bbox[2], bbox[3], class_id]
                    for bbox, class_id in zip(aug_bboxes, aug_class_ids)
                ]

                # Filtrar bboxes inválidas (fora da imagem ou dimensões inválidas)
                aug_bboxes_with_class = [
                    bbox for bbox in aug_bboxes_with_class
                    if 0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and bbox[2] > 0 and bbox[3] > 0
                ]

                # Salvar apenas se houver bboxes válidas ou se save_empty_labels=True
                if aug_bboxes_with_class or save_empty_labels:
                    aug_image_file = f"{os.path.splitext(image_file)[0]}_aug{i+1}{os.path.splitext(image_file)[1]}"
                    cv2.imwrite(os.path.join(output_image_dir, aug_image_file), aug_image)
                    aug_label_file = os.path.join(output_label_dir, f"{os.path.splitext(image_file)[0]}_aug{i+1}.txt")
                    if aug_bboxes_with_class:
                        write_yolo_label(aug_label_file, aug_bboxes_with_class)
                    elif save_empty_labels:
                        # Criar .txt vazio para imagens aumentadas sem bboxes
                        open(aug_label_file, 'a').close()

if __name__ == "__main__":
    input_dataset_dir = "./dataset_original"  # Caminho do dataset original
    output_dataset_dir = "./dataset"  # Caminho do dataset aumentado
    num_augmentations = 3  # Número de imagens aumentadas por imagem original
    save_empty_labels = False  # Não criar .txt vazios para imagens sem bboxes

    augment_dataset(input_dataset_dir, output_dataset_dir, num_augmentations, save_empty_labels)
    print(f"Dataset aumentado salvo em: {output_dataset_dir}")