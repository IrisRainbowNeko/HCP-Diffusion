import torchvision.transforms as T
from PIL import Image
from rainbowneko.data import DataHandler, HandlerChain, LoadImageHandler, ImageHandler

class ControlNetHandler(DataHandler):
    def __init__(self, key_map_in=('cond -> image',), key_map_out=('image -> cond',), bucket=None):
        super().__init__(key_map_in, key_map_out)

        self.handlers = HandlerChain(
            load=LoadImageHandler(),
            bucket=bucket.handler if bucket else DataHandler(),
            image=ImageHandler(
                transform=T.ToTensor(),
            )
        )

    def handle(self, image:Image.Image):
        return self.handlers(dict(image=image))