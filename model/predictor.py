import torch

from model.models import MixedModel, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT
from PIL import Image
from torchvision import transforms


class KeyPointPredictor:

    def __init__(self, model_file=None):
        self.model = MixedModel()
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

    def predict(self, image):

        width, height = image.size
        resized_image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        convert_tensor = transforms.ToTensor()
        resized_image_tensor = convert_tensor(resized_image)
        assert tuple(resized_image_tensor.shape) == (IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)

        resized_image_tensor = resized_image_tensor.view([1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])

        prediction = self.model(resized_image_tensor)

        print(prediction)