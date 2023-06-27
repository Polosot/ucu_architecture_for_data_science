from flask import Flask, request
from flask import render_template
from PIL import Image
from io import BytesIO
from model.predictor import KeyPointPredictor

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

        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run()
