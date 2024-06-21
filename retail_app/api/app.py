import os
import time
from functools import partial

import gradio as gr
import tensorflow as tf
from fastapi import FastAPI, Request
from prometheus_client import generate_latest
from ultralytics import YOLO

from src.inference import inference
from src.utils import record_request_time

app = FastAPI()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

detection_model = YOLO(os.environ["YOLO_MODEL_PATH"])

loaded = tf.saved_model.load(os.environ["OCR_MODEL_PATH"])
aocr_model = loaded.signatures["serving_default"]


@app.get("/metrics")
async def get_metrics():
    return generate_latest()


@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    try:
        # Ваша логика предсказания
        response = await demo(request)
        record_request_time(start_time, success=True)
        return response
    except Exception as e:
        record_request_time(start_time, success=False)
        raise e


if __name__ == "__main__":
    inference_foo = partial(
        inference, detection_model=detection_model, aocr_model=aocr_model
    )
    # Создание интерфейса Gradio
    demo = gr.Interface(
        fn=inference_foo,
        inputs="file",
        outputs=[
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
    demo.queue().launch(share=True, debug=True, app=app)
