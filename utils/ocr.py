import os
import sys
import pathlib

module_path = os.path.abspath(os.path.join('.'))

if module_path not in sys.path:
    sys.path.append(module_path)

from skimage.io import imread
from utils.apikeys import api_key
import io
import requests
import cv2
import json
import re


def get_text_from_image(filepath):
    image = imread(filepath)
    ftype = "."+filepath.parts[-1].split('.')[-1]
    _, compressed_image = cv2.imencode(ftype, image)
    image_bytes=io.BytesIO(compressed_image)
    ocr_endpoint_url = "https://api.ocr.space/parse/image"
    POST_params = {"apikey": api_key, "OCREngine" : 2}
    result = requests.post(url=ocr_endpoint_url,
    data=POST_params,
    files={"ocr_test.jpg" : image_bytes})
    result = result.content.decode()
    result = json.loads(result)
    text_in_image = result.get("ParsedResults")[0].get("ParsedText")
    print(text_in_image)
    text_in_image = re.sub(r"[^0-9]", '', str(text_in_image)) # remove any non-numeric characters
    text_in_image = re.sub(r"\s+", '', str(text_in_image)) # remove spaces
    print(text_in_image)
    return text_in_image
