from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class AirGuardDetector:

    def __init__(
        self,
        model_path: str = "models/best.pt"
    ):

        self.model_path = Path(
            model_path
        )

        self.model = None
        self.error = None


        # Load YOLO model
        if self.model_path.exists():

            try:

                from ultralytics import YOLO

                self.model = YOLO(
                    str(self.model_path)
                )

            except Exception as exc:

                self.error = str(exc)


    @property
    def ready(self) -> bool:

        return self.model is not None


    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.35
    ) -> list[dict[str, Any]]:

        if not self.ready:

            raise RuntimeError(
                "No compatible model found. "
                "Add your trained weights at "
                "models/best.pt."
            )


        frame = np.array(
            image.convert("RGB")
        )


        results = self.model.predict(
            source=frame,
            conf=confidence,
            verbose=False
        )


        detections = []


        for result in results:

            names = result.names


            if result.boxes is None:
                continue


            for box in result.boxes:

                class_id = int(
                    box.cls.item()
                )

                conf = float(
                    box.conf.item()
                )


                x1, y1, x2, y2 = [
                    float(value)
                    for value
                    in box.xyxy[0].tolist()
                ]


                detections.append({

                    "class":
                        str(names[class_id]),

                    "confidence":
                        conf,

                    "bbox":
                        [x1, y1, x2, y2]

                })


        return detections
