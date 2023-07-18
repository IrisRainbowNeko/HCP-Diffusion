import numpy as np
import torch
from PIL import Image
from diffusers.utils import PIL_INTERPOLATION

class ControlNetProcessor:
    def __init__(self, image):
        self.image_path = image

    def prepare_cond_image(self, image, width, height, batch_size, device):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32)/255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image = image.repeat_interleave(batch_size, dim=0)
        image = image.to(device=device)

        return image

    def __call__(self, width, height, batch_size, device, dtype):
        img = Image.open(self.image_path).convert('RGB')
        return self.prepare_cond_image(img, width, height, batch_size, 'cuda').to(dtype=dtype)
