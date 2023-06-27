from flask import Flask, request
from flask import render_template
from PIL import Image
from io import BytesIO
from model.predictor import KeyPointPredictor
from base64 import b64encode


app = Flask(__name__)
predictor = KeyPointPredictor()


@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        image = Image.open(BytesIO(f.read()))
        predictor.add_points(image)

        modified_image = BytesIO()
        image.save(modified_image, format='PNG')
        dataurl = 'data:image/png;base64,' + b64encode(modified_image.getvalue()).decode('ascii')

        return render_template('image.html', image_data=dataurl)


if __name__ == '__main__':
    app.run()
