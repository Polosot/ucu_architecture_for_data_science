import torch

from model.models import MixedModel, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import ImageDraw


class KeyPointPredictor:

    def __init__(self, model_file=None):
        self.model = MixedModel()
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

    def predict(self, image):

        if image.mode != 'L':
            image = image.convert('L')

        width, height = image.size

        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

        convert_tensor = transforms.ToTensor()
        resized_image_tensor = convert_tensor(image)
        assert tuple(resized_image_tensor.shape) == (IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)

        resized_image_tensor = resized_image_tensor.view([1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])

        prediction = self.model(resized_image_tensor)
        key_points = prediction.view([-1, 2]).detach().numpy() * np.array([[width, height]])

        return key_points

    def add_points(self, image):

        key_points = self.predict(image)
        drawer = ImageDraw.Draw(image)

        for p in key_points:
            drawer.rectangle(((p[0]-1, p[1]-1), (p[0]+1, p[1]+1)), fill="red")
