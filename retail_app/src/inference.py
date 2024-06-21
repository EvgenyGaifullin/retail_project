# from ultralytics import YOLO
import os
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.utils import draw_label, save_prediction_to_db

DATABASE_URL = os.environ["DATABASE_URL"]
LABELS = [
    "Штрихкод товара",
    "Название товара",
    "Цена без скидки, руб",
    "Цена без скидки, коп",
    "Акционная цена, руб",
    "Акционная цена, коп",
    "QR-код",
    "Номер товара на весах",
]


def aocr_inference(aocr, image):
    image = cv2.imencode(".png", image)[1].tobytes()
    tf_tensor = tf.convert_to_tensor(image)
    outputs = aocr(tf_tensor)
    price = outputs["output"].numpy()
    result = re.sub("[^0-9]", "", price.decode()[:-2])
    return result


def inference(image_file, detection_model, aocr_model):
    if len(image_file) > 1:
        raise Exception(
            "Пожалуйста, загрузите только одно изображение. В настоящий момент система не поддерживает обработку нескольких изображений."
        )
    image_path = image_file.name
    image = np.array(Image.open(image_path))
    results = detection_model(image)
    res_df = pd.DataFrame(
        columns=[
            "Название изображения",
            "Название класса",
            "Вероятность предсказания",
            "x_left",
            "y_top",
            "x_right",
            "y_bottom",
        ]
    )
    prices = {}
    result_image = image.copy()

    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes.xyxy):
            label, conf, coords = (
                int(boxes.cls[i].item()),
                boxes.conf[i].item(),
                list(map(int, box)),
            )
            x_left, y_top, x_right, y_bottom = coords
            res_df.loc[len(res_df)] = [
                image_path,
                LABELS[label],
                round(conf, 5),
                x_left,
                y_top,
                x_right,
                y_bottom,
            ]
            try:
                save_prediction_to_db(
                    image_path, LABELS[label], round(conf, 5), coords, DATABASE_URL
                )
            except Exception:
                print("Error writing recognition results to Database.")

            if label in [2, 3, 4, 5]:
                price_img = image[y_top:y_bottom, x_left:x_right]
                predicted_price = aocr_inference(aocr_model, price_img)

                if predicted_price:
                    prices[label] = predicted_price

            result_image = cv2.rectangle(
                result_image, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2
            )
            result_image = draw_label(result_image, label, x_left, y_top)

    # Format prices into a readable string
    source_price, promo_price = None, None
    if prices:
        source_price = f"{prices.get(2, '')} руб. {prices.get(3, '')} коп."
        promo_price = f"{prices.get(4, '')} руб. {prices.get(5, '')} коп."

    prices_df = pd.DataFrame(
        [{"Исходная Цена": source_price, "Акционная Цена (при наличии)": promo_price}]
    )

    return result_image, prices_df, res_df
