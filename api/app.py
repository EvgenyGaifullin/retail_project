import os
from functools import partial

import gradio as gr
import tensorflow as tf
from ultralytics import YOLO

from src.inference import inference

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

YOLO_MODEL_PATH = "/bristol/runs/detect/yolov8n_custom5/weights/best_openvino_model"
detection_model = YOLO(YOLO_MODEL_PATH)

OCR_MODEL_PATH = "/bristol/models/aocr-model"
loaded = tf.saved_model.load(OCR_MODEL_PATH)
aocr_model = loaded.signatures["serving_default"]


if __name__ == "__main__":
    
    inference_foo = partial(
        inference, detection_model=detection_model, aocr_model=aocr_model
    )
    # Создание интерфейса Gradio
    demo = gr.Interface(
        fn=inference_foo,
        inputs="file",
        outputs=[
            # "image",
            gr.Image(type="pil", label="label", height=400, width=400),
            gr.Dataframe(headers=["Исходная Цена", "Акционная Цена (при наличии)"]),
            gr.Dataframe(
                headers=[
                    "Название изображения",
                    "Название класса",
                    "Вероятность предсказания",
                    "x_left",
                    "y_top",
                    "x_right",
                    "y_bottom",
                ]
            ),
        ],
        title="Система распознавания цен",
        description="Загрузите одно или несколько изображений и сервис вернет Вам найденную информацию.",
    )
    # Запуск веб-сервиса
    demo.queue().launch(share=True, debug=True)
