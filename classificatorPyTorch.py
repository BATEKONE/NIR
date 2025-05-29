import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import gradio as gr

# ==== Кастомный датасет из YOLO ====
class YoloCropDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            img_name = fname.replace(".txt", ".jpg")
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, fname)

            if not os.path.exists(img_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split()
                    if len(items) != 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, items)
                    self.samples.append((img_path, class_id, x_center, y_center, width, height))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, x_center, y_center, width, height = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x1 = int(max(x_center - width / 2, 0))
        y1 = int(max(y_center - height / 2, 0))
        x2 = int(min(x_center + width / 2, w))
        y2 = int(min(y_center + height / 2, h))

        cropped = image.crop((x1, y1, x2, y2))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, int(class_id)

# ==== Трансформации ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Загрузка датасета ====
train_dataset = YoloCropDataset(
    image_dir="helmet_yolo_dataset/images/train",
    label_dir="helmet_yolo_dataset/labels/train",
    transform=transform
)

val_dataset = YoloCropDataset(
    image_dir="helmet_yolo_dataset/images/val",
    label_dir="helmet_yolo_dataset/labels/val",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ==== Модель ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# ==== Обучение ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 30

print("🚀 Начинаем обучение модели...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_acc += (outputs.argmax(1) == labels).sum().item()

    print(f"[{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}, "
          f"Train Acc: {correct/len(train_dataset):.2f}, Val Acc: {val_acc/len(val_dataset):.2f}")

# ==== Сохраняем модель ====
torch.save(model.state_dict(), "helmet_classifier.pth")
print("✅ Модель сохранена как helmet_classifier.pth")

# ==== Gradio-интерфейс с отрисовкой ====

def predict(image_np):
    model.eval()
    image = Image.fromarray(image_np).convert("RGB")
    w, h = image.size
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)

    label_dir = "helmet_yolo_dataset/labels/val"
    label_file = None

    # Попытка найти подходящий label
    image_name = None
    for fname in os.listdir(label_dir):
        possible_image = fname.replace(".txt", ".jpg")
        if possible_image in os.listdir("helmet_yolo_dataset/images/val"):
            image_name = possible_image
            label_file = os.path.join(label_dir, fname)
            break

    if label_file is None:
        return image_draw, "❌ Не найдена разметка YOLO для изображения"

    results = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_c, y_c, box_w, box_h = map(float, line.strip().split())

            x_c *= w
            y_c *= h
            box_w *= w
            box_h *= h
            x1 = int(max(x_c - box_w / 2, 0))
            y1 = int(max(y_c - box_h / 2, 0))
            x2 = int(min(x_c + box_w / 2, w))
            y2 = int(min(y_c + box_h / 2, h))

            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(crop_tensor)
                pred = output.argmax(1).item()

            label_text = "🟢 Helmet" if pred == 0 else "🔴 No helmet"
            color = "green" if pred == 0 else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 10), label_text, fill=color)

            results.append(label_text)

    return image_draw, ", ".join(results) if results else "🤷 Нет объектов"

# ==== Интерфейс Gradio ====

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Классификатор шлемов с визуализацией (YOLO + ResNet)")

    with gr.Row():
        input_image = gr.Image(label="📷 Входное изображение", type="numpy")
        output_image = gr.Image(label="🖼️ С результатом")
        output_text = gr.Label(label="🔍 Объекты")

    btn = gr.Button("Проверить")
    btn.click(fn=predict, inputs=input_image, outputs=[output_image, output_text])

    example_dir = "helmet_yolo_dataset/images/val"
    examples = [
        os.path.join(example_dir, f)
        for f in os.listdir(example_dir)
        if f.lower().endswith((".jpg", ".png"))
    ][:5]

    gr.Examples(examples=examples, inputs=input_image)

demo.launch()