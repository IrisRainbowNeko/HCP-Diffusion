import json
import os
import glob
import yaml

from loguru import logger


class BaseCaptionLoader:
    def __init__(self, path):
        self.path = path

    def _load(self):
        raise NotImplementedError
        
    def load(self):
        retval = self._load()
        logger.info(f'{len(retval)} record(s) loaded with {self.__class__.__name__}, from path {self.path!r}')
        return retval

class JsonCaptionLoader(BaseCaptionLoader):
    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

class YamlCaptionLoader(BaseCaptionLoader):
    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

class TXTCaptionLoader(BaseCaptionLoader):
    def _load(self):
        txt_files = glob.glob(os.path.join(self.path, '*.txt'))
        captions = {}
        for file in txt_files:
            with open(file, 'r', encoding='utf-8') as f:
                captions[os.path.basename(file).split('.')[0]] = f.read().strip()
        return captions

def auto_caption_loader(path):
    if os.path.isdir(path):
        json_files = glob.glob(os.path.join(path, '*.json'))
        if json_files:
            return JsonCaptionLoader(json_files[0])

        yaml_files = [
            *glob.glob(os.path.join(path, '*.yaml')),
            *glob.glob(os.path.join(path, '*.yml')),
        ]
        if yaml_files:
            return YamlCaptionLoader(yaml_files[0])

        txt_files = glob.glob(os.path.join(path, '*.txt'))
        if txt_files:
            return TXTCaptionLoader(path)

        raise FileNotFoundError(f'Caption file not found in directory {path!r}.')

    elif os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext == '.json':
            return JsonCaptionLoader(path)
        elif ext in {'.yaml', '.yml'}:
            return YamlCaptionLoader(path)
        else:
            raise FileNotFoundError(f'Unknown caption file {path!r}.')

    else:
        raise FileNotFoundError(f'Unknown caption file type {path!r}.')
