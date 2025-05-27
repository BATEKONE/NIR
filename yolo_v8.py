from ultralytics import YOLO
import gradio as gr
from PIL import Image
import cv2
import os

# Загрузка или обучение модели
def train_yolo_model():
    model = YOLO("yolov8n.pt")  # Можно использовать yolov8s.pt или yolov8m.pt для большей точности
    model.train(data="helmet_yolo_dataset/data.yaml", epochs=30, imgsz=640, batch=16)
    return model

# Если модель уже обучена
trained_model = YOLO("runs/detect/train/weights/best.pt")  # или загружаем обученную модель

def predict(image):
    # Конвертируем изображение в формат, подходящий для YOLO
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Получаем предсказания
    results = trained_model(img)[0]
    
    # Визуализируем результаты
    annotated_img = results.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Подсчитываем нарушения
    num_no_helmet = sum(box.cls == 1 for box in results.boxes)
    violation_text = f"Нарушения обнаружены: {num_no_helmet}" if num_no_helmet else "Нарушений не обнаружено"
    
    return Image.fromarray(annotated_img), violation_text

# Создаем интерфейс с Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Система обнаружения мотоциклистов без шлема (YOLOv8)")
    
    with gr.Row():
        input_image = gr.Image(label="Входное изображение")
        output_image = gr.Image(label="Результат обнаружения")
    
    violation_output = gr.Textbox(label="Результат проверки")
    
    submit_btn = gr.Button("Анализировать")
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, violation_output]
    )
    
    gr.Examples(
        examples=[os.path.join("test", f) for f in os.listdir("test") if f.endswith(".jpg")][:3],
        inputs=input_image
    )

demo.launch()