import os
import shutil
from typing import Dict

import tensorflow as tf
import yaml
from attention_ocr.aocr.model.model import Model
from attention_ocr.aocr.util.export import Exporter
from clearml import Dataset, Task, TaskTypes

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


def main(config):
    if not config["task_name"]:
        config["task_name"] = "ocr training"

    task = Task.init(
        project_name=config["clearml"]["project_name"],
        task_name=config["clearml"]["task_name"],
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
    )

    if config["clearml"]["tags"]:
        task.set_tags([*config["clearml"]["tags"]])

    if not config["training"]["aocr"]["data_preparation"]["dataset_path"].exists():
        download_dataset(config["training"]["aocr"]["data_preparation"])

    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    ) as sess:
        # training stage
        config["training"]["aocr"]["session"] = sess
        model = Model(**config["training"]["aocr"])
        model.train(
            data_path=config.data_preparation.output_path,
            num_epoch=config.global_variables.num_epochs,
        )
    tf.reset_default_graph()

    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    ) as sess:
        # model export stage
        sess.run(tf.global_variables_initializer())
        config["training"]["aocr"]["session"] = sess
        config["training"]["aocr"]["phase"] = "export"
        model = Model(**config["training"]["aocr"])
        exporter = Exporter(model)
        exporter.save(
            config["training"]["aocr"]["model_exporting"]["export_path"],
            config["training"]["aocr"]["model_exporting"]["export_format"],
        )


if __name__ == "__main__":
    with open("/app/configs/config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    main()
