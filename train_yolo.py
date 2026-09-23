from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")

    results = model.train(
        data=r"F:\Pycharm Projects\visual-novel-screen-translator\dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=2,
        name="vn_translator_yolo26",
        resume=False,
    )