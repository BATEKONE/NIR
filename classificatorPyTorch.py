import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import gradio as gr
from PIL import Image
import cv2
import numpy as np

# Загрузка модели
def load_classifier_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 класса: helmet и no_helmet
    model.load_state_dict(torch.load("helmet_classifier.pth"))
    model.eval()
    return model

model = load_classifier_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Трансформации изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция для предсказания
def predict(image):
    # Преобразуем изображение
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)
    
    # Получаем предсказание
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    # Возвращаем результат
    return "Шлем надет" if preds[0] == 0 else "Шлем не надет!"

# Gradio интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Классификатор наличия шлема")
    
    with gr.Row():
        input_image = gr.Image(label="Изображение мотоциклиста")
        output_label = gr.Label(label="Результат классификации")
    
    submit_btn = gr.Button("Проверить")
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )
    
    gr.Examples(
        examples=[os.path.join("test_crops", f) for f in os.listdir("test_crops") if f.endswith(".jpg")][:5],
        inputs=input_image
    )

demo.launch()