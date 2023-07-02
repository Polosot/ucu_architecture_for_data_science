from base64 import b64encode
from io import BytesIO

import imageio.v3 as iio
import numpy as np
from PIL import Image
from flask import Flask, request
from flask import render_template

from model.predictor import KeyPointPredictor

app = Flask(__name__)
predictor = KeyPointPredictor(model_file='artifacts/mixed_model_weights.pkl')


@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)


@app.route('/image_uploader', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        image = Image.open(BytesIO(f.read()))
        predictor.add_points(image)

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
