from ultralytics import YOLO
import gradio as gr
from PIL import Image
import cv2
import os
import torch

print("[INFO] CUDA доступен:", torch.cuda.is_available())
print("[INFO] Текущее устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Пути
MODEL_PATH = "runs/detect/train/weights/best.pt"
DATA_YAML_PATH = "helmet_yolo_dataset/data.yaml"
YOLO_ARCH = "yolov8s.pt"  # можно заменить на yolov8s.pt для ускорения

# Проверка наличия обученной модели
if not os.path.exists(MODEL_PATH):
    print("[INFO] Модель не найдена. Начинаем обучение YOLOv8...")
    model = YOLO(YOLO_ARCH)
    model.train(data=DATA_YAML_PATH, epochs=30, imgsz=640, batch=16, device='cuda', workers=0)
    trained_model = model
    print("[INFO] Обучение завершено.")
else:
    print("[INFO] Загружается обученная модель...")
    trained_model = YOLO(MODEL_PATH)

# Предсказание
def predict(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = trained_model(img)[0]
    annotated_img = results.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    num_no_helmet = sum(int(box.cls) == 1 for box in results.boxes)
    violation_text = f"Нарушения обнаружены: {num_no_helmet}" if num_no_helmet else "Нарушений не обнаружено"

    return Image.fromarray(annotated_img), violation_text

# Интерфейс Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🪖 Система обнаружения мотоциклистов без шлема (YOLOv8)")

    with gr.Row():
        input_image = gr.Image(label="Входное изображение")
        output_image = gr.Image(label="Результат обнаружения")

    violation_output = gr.Textbox(label="Результат проверки")
    submit_btn = gr.Button("Анализировать")

    submit_btn.click(fn=predict, inputs=input_image, outputs=[output_image, violation_output])

    if os.path.exists("test"):
        examples = [os.path.join("test", f) for f in os.listdir("test") if f.lower().endswith((".jpg", ".png"))][:3]
        gr.Examples(examples=examples, inputs=input_image)

demo.launch()