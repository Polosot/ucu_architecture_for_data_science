
#### Ukrainian Catholic University
### Software Architecture for Data Science in Python

# Study project: Facial point detector

#### Student: Maksym Sarana

## Branches:

- hw1_mvp - MVP

- hw2_final - final version

## Run application

There are two options, run as a docker container or as a Python application:

### Python run:

1. Clone git repo:
 
```
git clone git@github.com:Polosot/ucu_architecture_for_data_science.git
cd ucu_architecture_for_data_science
```

2. Setup virtual environment:
```
python -m venv  env
source env/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Install `ffmpeg` (for video processing)
 
Ubuntu: ```sudo apt install ffmpeg```

MacOS: ```brew install ffmpeg```
 
Or download / follow instructions from https://ffmpeg.org/

5. Run application:
```
python app.py
```

### Docker run:

1. Clone git repo:
 
```
git clone git@github.com:Polosot/ucu_architecture_for_data_science.git
cd ucu_architecture_for_data_science
```

2. Build the Docker image
```
docker build -t facepoints .
```

3. Run the Docker container

```
docker run -p 5000:5000 --name facepoints_container facepoints
```

## How to use the application

1. Open: http://localhost:5000

2. Choose a file from the [test_media](test_media) subdirectory and submit.

## Changes of the final version:

1. Detect faces with `cv2` and crop them for the predictor
2. Possibility to work with multiple faces
3. Work with video
4. Model selection (4 and 15 key points)
5. Dockerfile
