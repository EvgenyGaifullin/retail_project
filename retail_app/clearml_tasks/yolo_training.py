import yaml
from clearml import Task, TaskTypes
from ultralytics import YOLO


def run_task(clearml_config, training_config):
    task = Task.init(
        project_name=clearml_config["project_name"],
        task_name=clearml_config["task_name"],
        task_type=TaskTypes.training,
        reuse_last_task_id=False,
        # auto_connect_frameworks={"hydra": False},
    )
    # Load the model.
    model = YOLO(f"{training_config["model_name"]}.pt")

    # Training.
    model.train(
        data=training_config["dataset_path"],
        imgsz=training_config["dataset_path"],
        epochs=training_config["epochs"],
        batch=training_config["batch"],
        device=[training_config["device"]],
        name=f"{training_config["model_name"]}_custom",
    )


if __name__ == "__main__":
    with open("/app/configs/config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    clearml_config = config["clearml"]
    training_config = config["training"]["yolo"]
    run_task(clearml_config, training_config)
