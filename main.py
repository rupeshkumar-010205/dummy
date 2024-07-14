import cv2
from main_parser import parse_args
from model.predict import DetectionPredictor
from model.train import DetectionTrainer
from model.validator import DetectionValidator


def train(overrides):
    trainer=DetectionTrainer(cfg=overrides)
    trainer.train()
def val(overrides):
    validator=DetectionValidator(args=overrides)
    validator()
def predict_source(overrides,model=r"runs\detect\train\weights\best.pt",source=r"assets\bus.jpg"):
    model=model if "model" not in overrides else overrides["model"]
    source=source if "source" not in overrides else overrides["source"]
    predictor=DetectionPredictor(cfg=overrides)
    predictor.setup_model(model=model, verbose=True)
    results=predictor(source=source,stream=False)
    for i in results:
        pass

def webcam(overrides,source=0):
    # Open the video file
    cap = cv2.VideoCapture(source)
    overrides["mode"]="predict"
    overrides["save"]=False
    predictor=DetectionPredictor(cfg=overrides)
    predictor.setup_model(model=overrides["model"], verbose=True)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = predictor(frame)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed or window is closed
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('YOLOv8 Inference', cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
def main(overrides):
    if overrides["mode"]=="val":
        overrides.pop("webcam")
        val(overrides)
    elif overrides["mode"]=="predict":
        if overrides["webcam"]:
            overrides.pop("webcam")
            webcam(overrides)
        else:
            overrides.pop("webcam")
            predict_source(overrides=overrides)
    else:
        overrides.pop("webcam")
        train(overrides)

if __name__ == "__main__":
    overrides = parse_args()
    overrides=vars(overrides)
    main(overrides)