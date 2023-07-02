import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms

from model.models import MixedModel, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT

MARKER_RELATIVE_HALF_SIZE = 0.005
MARKER_COLOR = "yellow"


class KeyPointPredictor:

    def __init__(self, model_file=None):
        self.model = MixedModel()
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.convert_tensor = transforms.ToTensor()

    def predict(self, gs_image, small_model=False):
        """
        Predict key points on the image
        Args:
            gs_image: PIL Image
                A greyscale image of a face (square form is preferred)
            small_model: boolean
                If true, only 4 key points are predicted
        Returns:
            list of tuples in format [(x, y), ...]
        """
        # resize the image if necessary
        width, height = gs_image.size
        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            gs_image = gs_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

        # convert to a proper pyTorch tensor
        resized_image_tensor = self.convert_tensor(gs_image)
        assert tuple(resized_image_tensor.shape) == (IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
        resized_image_tensor = resized_image_tensor.view([1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])

        # predict key points (in coordinates 0..1)
        model = self.model.stage_1_model if small_model else self.model
        prediction = model(resized_image_tensor)

        # adjust key points to the original image size
        key_points = prediction.view([-1, 2]).detach().numpy() * np.array([[width, height]])

        return key_points

    def detect_faces(self, gs_image):
        """
        The function detects faces on the image with the cv2 function
        Args:
            gs_image: PIL Image
                A greyscale image
        Returns:
            list of tuples in the format [(x, y, w, h), ...]
        """
        faces = self.face_classifier.detectMultiScale(
            np.array(gs_image), scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
        )

        return faces

    def detect_key_points(self, gs_image, bounding_box, **kwargs):
        """
        The function predicts key points for one face on the greyscale image
        Args:
            gs_image: PIL image
                Greyscale image
            bounding_box: tuple
                Bounding box in the format (x, y, w, h)
            **kwargs:
                Additional parameters
        Returns:
            list of key points in the format [(x, y), ...]
        """
        # crop an image
        x, y, w, h = bounding_box
        im_face = gs_image.crop((x, y, x+w, y+h))

        # predict key points
        key_points = self.predict(im_face, **kwargs)

        # adjust key point to the original image size
        return [(key_point[0]+x, key_point[1]+y) for key_point in key_points]

    def find_keypoints(self, image, **kwargs):
        """
        The function predicts all key points on the image
        Args:
            image: PIL Image
                An image of arbitrary size and number of channels
            **kwargs:
                Additional parameters
        Returns:
            list of tuples
                Key points in the format [(x, y), ...]
        """
        # convert to greyscale
        gs_image = image.convert('L') if image.mode != 'L' else image

        # detect faces
        faces = self.detect_faces(gs_image)

        # find key points for each of the detected faces
        key_points = []
        for face in faces:
            key_points.extend(self.detect_key_points(gs_image, bounding_box=face, **kwargs))

        return key_points

    def draw_keypoints(self, image, key_points):
        """
        The function draws the given key points on the image
        Args:
            image: PIL Image
                An image of arbitrary size and number of channels
            key_points: list of tuples
                A list of key points [(x, y), ...]
        """

        drawer = ImageDraw.Draw(image)
        marker_half_size = max(1, int(min(image.size) * MARKER_RELATIVE_HALF_SIZE))
        for p in key_points:
            drawer.rectangle(
                ((p[0]-marker_half_size, p[1]-marker_half_size), (p[0]+marker_half_size, p[1]+marker_half_size)),
                fill=MARKER_COLOR
            )

    def process_image(self, image, **kwargs):
        """
        The function predicts and adds facial points to the image
        Args:
            image: PIL Image
                An image of arbitrary size and number of channels
            **kwargs:
                Additional parameters
        """

        key_points = self.find_keypoints(image, **kwargs)
        self.draw_keypoints(image, key_points)

    def process_frames(self, frames, **kwargs):
        """
        The function predicts and adds facial points to the sequence of frames
        Args:
            frames: np.ndarray
                Frames in the format (frames, channels, height, width)
            **kwargs:
                Additional parameters
        Returns:
            a generator object of a PIL Image
        """

        for frame in frames:
            image = Image.fromarray(frame)
            self.process_image(image, **kwargs)
            yield image

