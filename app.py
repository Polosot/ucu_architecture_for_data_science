from flask import Flask, request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']

      return 'file uploaded successfully'


if __name__ == '__main__':
    app.run()