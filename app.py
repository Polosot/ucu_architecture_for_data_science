from base64 import b64encode
from io import BytesIO

import imageio.v3 as iio
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from enum import Enum

from model.predictor import KeyPointPredictor

MIN_IMAGE_SIZE = 200
MAX_IMAGE_SIZE = 1000

class MediaType(Enum):
    IMAGE = 1
    VIDEO = 2


app = Flask(__name__, static_folder="static")
predictor = KeyPointPredictor(model_file='artifacts/mixed_model_weights.pkl')


def get_image_shape(h, w):

    min_size = min(h, w)
    max_size = max(h, w)

    if min_size <= MIN_IMAGE_SIZE:
        scale = MIN_IMAGE_SIZE / min_size
        return int(h * scale), int(w * scale)
    elif max_size >= MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max_size
        return int(h * scale), int(w * scale)
    else:
        return h, w


def process_image(f, small_model=False):
    image = Image.open(BytesIO(f.read()))
    h, w = get_image_shape(image.height, image.width)

    predictor.process_image(image, small_model=small_model)

    modified_media_bytes = BytesIO()
    image.save(modified_media_bytes, format='PNG')
    output_mimetype = 'image/png'
    return f"data:{output_mimetype};base64,{b64encode(modified_media_bytes.getvalue()).decode('ascii')}", h, w


def process_video(f, small_model=False):
    fb = f.read()
    extension = '.' + f.mimetype.split('/', maxsplit=1)[-1]
    frames = iio.imread(fb, extension=extension)
    fps = iio.immeta(fb, extension=extension)['fps']

    frames_changed = np.array(
        [np.array(image) for image in predictor.process_frames(frames, small_model=small_model)])
    modified_media_bytes = iio.imwrite('<bytes>', frames_changed, extension='.mp4', fps=fps)
    output_mimetype = 'video/mp4'
    return f"data:{output_mimetype};base64,{b64encode(modified_media_bytes).decode('ascii')}"


@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)


@app.route('/file_uploader', methods=['GET', 'POST'])
def submit_media():
    f = request.files['file']
    small_model = (request.form.get('model_selector') == 'small')

    if f.mimetype.startswith('image/'):
        output_media, h, w = process_image(f, small_model)
        return render_template('image.html', image_data=output_media, image_height=h, image_width=w)

    elif f.mimetype.startswith('video/'):
        output_media = process_video(f, small_model)
        return render_template('video.html', video_data=output_media)

    else:
        raise NotImplementedError(f"Mimetype {f.mimetype} is not implemented")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
