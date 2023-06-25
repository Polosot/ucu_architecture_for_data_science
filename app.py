from flask import Flask
from flask import render_template

server = Flask(__name__)

@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)

if __name__ == '__main__':
    server.run()