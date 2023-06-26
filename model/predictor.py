import torch

from models import MixedModel


class KeyPointPredictor:

    def __init__(self, model_file=None):
        self.model = MixedModel()
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

    def predict(self, image):
        pass