import base64
import io
import os
import json
import threading

import requests
from PIL import Image

from utils import storage

REST_URL = os.environ.get('REST_URL')
PREDICT_URL = f"{REST_URL}/v1/models/model:predict"


class ImageGenerator:
    def __init__ (self, prediction_id, image_id, prompt):
        self.prediction_id = prediction_id
        self.image_id = image_id
        self.prompt = prompt
        self.image_json = {
            "id": self.prediction_id,
            "status": "QUEUED",
            "progress": 0,
            "prompt": prompt,
            "file": f"/api/images/{self.prediction_id}/image-{self.image_id}.jpg"
        }
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def write_image_json(self, status, progress):
        if status:
            self.image_json["status"] = status
        if progress is not None:
            self.image_json["progress"] = progress

        json_file = os.path.join(self.prediction_id, f"image-{self.image_id}.json")
        storage.write_json(self.image_json, json_file)

    def create_image(self):
        print("image_request")
        json_data = {
            "instances": [
                {
                    "prompt": self.image_json["prompt"],
                    # "negative_prompt": "",
                    # "num_inference_steps": 60,
                }
            ]
        }

        response = requests.post(PREDICT_URL, json=json_data, verify=False)
        response_dict = response.json()

        img_str = response_dict["predictions"][0]["image"]["b64"]
        img_data = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_data))
        storage.write_image(image, os.path.join(self.prediction_id, f"image-{self.image_id}.jpg"))

    def run(self):
        print(f"Running image generator {self.prediction_id}/image-{self.image_id} on prompt {self.prompt}")
        self.write_image_json("IN_PROGRESS", 0)
        self.create_image()
        self.write_image_json("COMPLETE", 100)


