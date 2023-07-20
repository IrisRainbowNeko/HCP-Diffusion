import json
import os
import glob
import yaml

class BaseCaptionLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        raise NotImplementedError()

class JsonCaptionLoader(BaseCaptionLoader):
    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

class YamlCaptionLoader(BaseCaptionLoader):
    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

class TXTCaptionLoader(BaseCaptionLoader):
    def load(self):
        txt_files = os.listdir(self.path)
        captions = {}
        for file in txt_files:
            with open(os.path.join(self.path, file), 'r', encoding='utf-8') as f:
                captions[file] = f.read().strip()
        return captions

def auto_caption_loader(path):
    if len(glob.glob('*.json'))>0:
        return JsonCaptionLoader(path)
    elif len(glob.glob('*.yaml'))>0:
        return YamlCaptionLoader(path)
    elif len(glob.glob('*.txt'))>0:
        return TXTCaptionLoader(path)
    else:
        raise FileNotFoundError('Caption file not found')