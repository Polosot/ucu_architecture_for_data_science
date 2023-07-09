from base64 import b64encode
from io import BytesIO

import imageio.v3 as iio
import numpy as np
from PIL import Image
from flask import Flask, request, render_template

from model.predictor import KeyPointPredictor

app = Flask(__name__, static_folder="static")
predictor = KeyPointPredictor(model_file='artifacts/mixed_model_weights.pkl')


def submit_media(small_model=False):
    f = request.files['file']
    if f.mimetype.startswith('image/'):
        image = Image.open(BytesIO(f.read()))

        predictor.process_image(image)

        modified_media_bytes = BytesIO()
        image.save(modified_media_bytes, format='PNG')
        output_mimetype = 'image/png'
    elif f.mimetype.startswith('video/'):
        fb = f.read()
        extension = '.' + f.mimetype.split('/', maxsplit=1)[-1]
        frames = iio.imread(fb, extension=extension)
        fps = iio.immeta(fb, extension=extension)['fps']

        frames_changed = np.array(
            [np.array(image) for image in predictor.process_frames(frames, small_model=small_model)])
        modified_media_bytes = iio.imwrite('<bytes>', frames_changed, extension='.mp4', fps=fps)
        output_mimetype = 'video/mp4'
    else:
        raise NotImplementedError(f"Mimetype {f.mimetype} is not implemented")

    return f"data:{output_mimetype};base64,{b64encode(modified_media_bytes.getvalue()).decode('ascii')}"


@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)


@app.route('/file_uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        image = Image.open(BytesIO(f.read()))
        predictor.process_image(image)

        modified_image = BytesIO()
        image.save(modified_image, format='PNG')
        dataurl = 'data:image/png;base64,' + b64encode(modified_image.getvalue()).decode('ascii')

        return render_template('image.html', image_data=dataurl)


@app.route('/video_uploader', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['file']
        if f.mimetype.startswith('video/'):
            extension = '.' + f.mimetype.split('/', maxsplit=1)[-1]

            fb = f.read()
            frames = iio.imread(fb, extension=extension)
            fps = iio.immeta(fb, extension=extension)['fps']

            frames_changed = np.array([np.array(image) for image in predictor.process_frames(frames, small_model=False)])
            fb_changed = iio.imwrite('<bytes>', frames_changed, extension='.mp4', fps=fps)  #TODO: extension?

            dataurl = 'data:video/mp4;base64,' + b64encode(fb_changed).decode('ascii')

            return render_template('video.html', video_data=dataurl)
        else:
            return "Error: You should submit a video file"


if __name__ == '__main__':
    app.run()
