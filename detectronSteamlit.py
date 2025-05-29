import streamlit as st
import numpy as np
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Настройка конфигурации Detectron2
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

# Создание предиктора
cfg = setup_cfg()
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Streamlit интерфейс
st.title("Система обнаружения мотоциклистов без шлема (Detectron2)")

uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Чтение изображения
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Получение предсказаний
    outputs = predictor(image)
    
    # Визуализация результатов
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Подсчет нарушений
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    num_no_helmet = sum(pred_classes == 1)
    
    # Отображение результатов
    st.image(out.get_image()[:, :, ::-1], caption="Результат обнаружения", use_column_width=True)
    
    if num_no_helmet > 0:
        st.error(f"Обнаружены нарушения: {num_no_helmet} мотоциклистов без шлема!")
    else:
        st.success("Нарушений не обнаружено")