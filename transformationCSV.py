import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import yaml

def prepare_dataset(csv_path, images_dir, output_dir):
    # Создаем папки для YOLO формата
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
        
    # Читаем CSV файл
    df = pd.read_csv(csv_path)
    
    # Группируем по изображениям
    grouped = df.groupby('image_id')
    
    # Список всех изображений
    all_images = list(grouped.groups.keys())
    
    # Разделяем на train/val
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Функция для обработки одного изображения
    def process_image(img_name, mode):
        # Получаем размеры изображения
        img_path = os.path.join(images_dir, f"{img_name}.jpg")
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except FileNotFoundError:
            print(f"Image {img_name} not found, skipping...")
            return
        
        # Копируем изображение в соответствующую папку
        shutil.copy(img_path, f"{output_dir}/images/{mode}/{img_name}.jpg")
        
        # Создаем файл с аннотациями
        label_file = f"{output_dir}/labels/{mode}/{img_name}.txt"
        with open(label_file, 'w') as f:
            for _, row in grouped.get_group(img_name).iterrows():
                # Преобразуем метку
                label = 0 if 'Helmet' in row['label'] else 1
                
                # Нормализуем координаты для YOLO
                x_center = ((row['x1'] + row['x2']) / 2) / width
                y_center = ((row['y1'] + row['y2']) / 2) / height
                box_width = (row['x2'] - row['x1']) / width
                box_height = (row['y2'] - row['y1']) / height
                
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
    
    # Обрабатываем train и val изображения
    for img in train_images:
        process_image(img, 'train')
    
    for img in val_images:
        process_image(img, 'val')
    
    # Создаем YAML файл конфигурации
    data_yaml = {
        'train': f'{output_dir}/images/train',
        'val': f'{output_dir}/images/val',
        'nc': 2,
        'names': ['helmet', 'no_helmet']
    }
    
    with open(f'{output_dir}/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

# Пример использования
prepare_dataset(
    csv_path="helmet_detection_train.csv",
    images_dir="images",
    output_dir="helmet_yolo_dataset"
)