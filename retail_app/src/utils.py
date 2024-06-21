import os
import shutil
import time
from typing import Dict

import cv2
import psycopg2
from clearml import Dataset
from prometheus_client import Counter, Summary, start_http_server


def download_dataset(config: Dict[str, str]) -> str:
    os.makedirs(config["dataset_path"], exist_ok=True)
    if len(os.listdir(config["dataset_path"])) > 0:
        shutil.rmtree(config["dataset_path"])
        os.makedirs(config["dataset_path"], exist_ok=True)

    dataset = Dataset.get(dataset_id=config["dataset_clearml_id"])
    dataset_path = dataset.get_mutable_local_copy(
        target_folder=config["dataset_path"], overwrite=False
    )
    return dataset_path


def draw_label(image, label, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y)
    fontScale = 1
    fontColor = (0, 0, 200)
    thickness = 2
    cv2.putText(
        image,
        str(label),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
    )
    return image


# Функция для сохранения данных в базу
def save_prediction_to_db(image_name, class_id, probability, box_coordinates, db_url):
    x_left, y_top, x_right, y_bottom = box_coordinates
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (filename, label, probability, x_left, y_top, x_right, y_bottom ) VALUES (%s, %i, %f, %i, %i, %i, %i)",
                    (
                        image_name,
                        class_id,
                        probability,
                        x_left,
                        y_top,
                        x_right,
                        y_bottom,
                    ),
                )
    except psycopg2.Error as e:
        print("Ошибка при подключении к базе данных:", e)


# Создаем метрики
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
REQUEST_COUNTER = Counter("request_count", "Total request count")
SUCCESSFUL_REQUESTS = Counter(
    "successful_request_count", "Total successful request count"
)
FAILED_REQUESTS = Counter("failed_request_count", "Total failed request count")

# Запускаем сервер метрик на порту 8000
start_http_server(8000)


def record_request_time(start_time, success=True):
    duration = time.time() - start_time
    REQUEST_TIME.observe(duration)
    REQUEST_COUNTER.inc()
    if success:
        SUCCESSFUL_REQUESTS.inc()
    else:
        FAILED_REQUESTS.inc()
