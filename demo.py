import random

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

object_tracker = DeepSort(
    n_init=3,
    max_age=30,
    nms_max_overlap=1.0,
    max_cosine_distance=0.5,
    max_iou_distance=0.5,
    embedder="mobilenet",
    # half=True,
    bgr=True,
    embedder_gpu=False,
    nn_budget=None,
    gating_only_position=True,
)

model = YOLO("./models/yolov8n.pt")
classes = model.names


colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for j in range(10)
]
cap = cv2.VideoCapture(
    "/Users/3i-a1-2021-15/Developer/projects/pivo-tracking/videos/2.mp4"
)
ret, image = cap.read()

cap_out = cv2.VideoWriter(
    "./results/out2.mp4",
    cv2.VideoWriter_fourcc(*"MP4V"),
    cap.get(cv2.CAP_PROP_FPS),
    (image.shape[1], image.shape[0]),
)

while ret:
    results = model(image)

    for result in results:
        detections = []
        boxes = result.boxes
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2 = r[:4]
            w, h = x2 - x1, y2 - y1
            coordinates = list((int(x1), int(y1), int(w), int(h)))
            conf = r[4]
            clsId = int(r[5])
            cls = classes[clsId]
            if cls == "person" and conf > 0.7:
                detections.append((coordinates, conf, cls))

        print("detections: ", detections)
        tracks = object_tracker.update_tracks(detections, frame=image)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            color = colors[int(track_id) % len(colors)]
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                thickness=4,
            )

            display_text = "ID: " + str(track_id)
            (text_width, text_height), _ = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
            )

            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1]) - 30),
                (
                    int(bbox[0]) + int(text_width),
                    int(bbox[1]),
                ),
                color,
                -1,
            )

            text_x = int(bbox[0]) + 5
            text_y = int(bbox[1]) + text_height + 5
            cv2.putText(
                image,
                display_text,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", image)
    cap_out.write(image)
    ret, image = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cap_out.release()
cv2.destroyAllWindows()
