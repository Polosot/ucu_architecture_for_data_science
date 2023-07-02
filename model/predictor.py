import torch

from model.models import MixedModel, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import ImageDraw
import cv2


class KeyPointPredictor:

    def __init__(self, model_file=None):
        self.model = MixedModel()
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

    def predict(self, gs_image, small_model=False):

        self.detect_faces(gs_image)

        width, height = gs_image.size

        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            gs_image = gs_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

        convert_tensor = transforms.ToTensor()
        resized_image_tensor = convert_tensor(gs_image)
        assert tuple(resized_image_tensor.shape) == (IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)

        resized_image_tensor = resized_image_tensor.view([1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])

        prediction = self.model.stage_1_model(resized_image_tensor) if small_model else self.model(resized_image_tensor)
        key_points = prediction.view([-1, 2]).detach().numpy() * np.array([[width, height]])

        return key_points

    def detect_faces(self, gs_image):
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_classifier.detectMultiScale(
            np.array(gs_image), scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
        )

        return faces

    def add_points(self, image, add_bb=True):
        # convert to greyscale
        gs_image = image.convert('L') if image.mode != 'L' else image

        # detect faces
        faces = self.detect_faces(gs_image)

        # predict key points
        key_points = []
        bboxes = []
        for face in faces:
            x, y, w, h = face
            im_face = gs_image.crop((x, y, x+w, y+h))
            face_key_points = self.predict(im_face)
            for kp in face_key_points:
                key_points.append((kp[0]+x, kp[1]+y))

            if add_bb:
                bboxes.append((x, y, x+w, y+h))

        drawer = ImageDraw.Draw(image)
        marker_half_size = int(min(image.size) * 0.005)
        for p in key_points:
            drawer.rectangle(((p[0]-marker_half_size, p[1]-marker_half_size), (p[0]+marker_half_size, p[1]+marker_half_size)), fill="yellow")

        for bb in bboxes:
            drawer.rectangle(bb, fill=None, outline="red", width=3)

    def process_frames(self, frames):

        for i in range(frames.shape[0]):
            image = Image.fromarray(frames[i])
            self.add_points(image)
            yield image

