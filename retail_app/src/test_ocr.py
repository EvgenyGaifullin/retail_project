import os
import re
import time
import warnings
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import tensorflow as tf
from clearml import Task, TaskTypes
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def convert_coords(labels_raw, img_shape):
    dh, dw, _ = img_shape
    x, y, w, h = map(float, labels_raw)

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    return (l, t), (r, b)


def aocr_inference(aocr, image):
    image = cv2.imencode(".png", image)[1].tobytes()
    tf_tensor = tf.convert_to_tensor(image)
    outputs = aocr(tf_tensor)
    price = outputs["output"].numpy()
    result = re.sub("[^0-9]", "", price.decode()[:-2])
    return result


def easyocr_inference(reader, image):
    result = reader.readtext(image, detail=0)
    result = " ".join(result)
    result = re.sub("[^0-9]", "", result)
    return result


def trocr_inference(model, processor, image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    trocr_price = re.sub("[^0-9]", "", generated_text)
    return trocr_price


ocr_model_path = "/bristol/models/aocr-model"
loaded = tf.saved_model.load(ocr_model_path)
aocr_model = loaded.signatures["serving_default"]

reader = easyocr.Reader(
    ["en"],
    gpu=False,
)

trocr_small_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-printed", use_fast=False
)
trocr_small = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

trocr_base_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed", use_fast=False
)
trocr_base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

annotations = list(Path("/bristol/data").glob("*/labels/*.txt"))

if __name__ == "__main__":
    task = Task.init(
        project_name="retail",
        task_name="ocr_inference",
        task_type=TaskTypes.inference,
        reuse_last_task_id=False,
        # auto_connect_frameworks={"hydra": False},
    )
    res = pd.DataFrame()
    for i in tqdm(range(len(annotations))):
        img = Image.open(
            str(f"{annotations[i].parent / annotations[i].stem}.jpg").replace(
                "labels", "images"
            )
        )
        img = np.array(img)

        labels_info = open(annotations[i]).read().splitlines()

        for line in labels_info:
            line_info = line.split(" ")
            label = int(line_info[0])
            if label in [2, 3, 4, 5]:
                left_top, right_bottom = convert_coords(line_info[1:], img.shape)
                label_crop = img[
                    left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]
                ]
                aocr_start = time.time()
                aocr_pred = aocr_inference(aocr_model, label_crop)
                aocr_end = time.time() - aocr_start

                trocr_small_start = time.time()
                trocr_small_pred = trocr_inference(
                    trocr_small, trocr_small_processor, label_crop
                )
                trocr_small_end = time.time() - trocr_small_start

                trocr_base_start = time.time()
                trocr_base_pred = trocr_inference(
                    trocr_base, trocr_base_processor, label_crop
                )
                trocr_base_end = time.time() - trocr_base_start

                easy_start = time.time()
                easyocr_pred = easyocr_inference(reader, label_crop)
                easy_end = time.time() - easy_start
                res.loc[
                    len(res),
                    [
                        "filename",
                        "label",
                        "trocr_base",
                        "aocr",
                        "trocr_small",
                        "easyocr",
                        "trocr_b_time",
                        "aocr_time",
                        "trocr_s_time",
                        "easy_time",
                    ],
                ] = [
                    str(annotations[i]),
                    label,
                    trocr_base_pred,
                    aocr_pred,
                    trocr_small_pred,
                    easyocr_pred,
                    trocr_base_end,
                    aocr_end,
                    trocr_small_end,
                    easy_end,
                ]
    res.to_csv("/bristol/ocr_res.csv", index=False)
