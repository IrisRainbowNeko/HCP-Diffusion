"""
train_ac.py
====================
    :Name:        convert old cfg format to new format
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

from omegaconf import ListConfig, DictConfig, OmegaConf

class DatasetCFGConverter:

    def convert_source(self, cfg_source:DictConfig):
        if '_target_' not in cfg_source:
            cfg_source['_target_'] = 'hcpdiff.data.source.Text2ImageAttMapSource'

        if 'tag_transforms' in cfg_source:
            cfg_source['text_transforms'] = cfg_source.pop('tag_transforms')

    def convert(self, cfg:DictConfig):
        for dataset in cfg['data'].values():
            for source in dataset['source'].values():
                self.convert_source(source)
        return cfg

class TrainCFGConverter:
    def __init__(self):
        self.dataset_converter = DatasetCFGConverter()

    def convert_model(self, cfg_model:DictConfig):
        if 'ema_unet' in cfg_model and 'ema' not in cfg_model:
            if cfg_model['ema_unet']==0: # no ema
                cfg_model['ema'] = None
            else:
                cfg_model['ema'] = OmegaConf.create({
                    '_target_': 'hcpdiff.utils.ema.ModelEMA',
                    '_partial_': True,
                    'decay_max': cfg_model['ema_unet'],
                    'power': 0.85
                })

        if 'tokenizer' not in cfg_model:
            cfg_model['tokenizer'] = None
        if 'noise_scheduler' not in cfg_model:
            cfg_model['noise_scheduler'] = None
        if 'unet' not in cfg_model:
            cfg_model['unet'] = None
        if 'text_encoder' not in cfg_model:
            cfg_model['text_encoder'] = None
        if 'vae' not in cfg_model:
            cfg_model['vae'] = None

    def convert_loss(self, cfg_train:DictConfig):
        if cfg_train['loss']['criterion']['_target_']=='hcpdiff.loss.MSELoss':
            cfg_train['loss']['criterion']['_target_'] = 'torch.nn.MSELoss'

    def convert(self, cfg:DictConfig):
        self.convert_model(cfg['model'])
        self.convert_loss(cfg['train'])

        if 'previewer' not in cfg:
            cfg['previewer'] = None

        cfg = self.dataset_converter.convert(cfg)
        return cfg

class InferCFGConverter:

    def convert(self, cfg:DictConfig):
        if 'encoder_attention_mask' not in cfg:
            cfg['encoder_attention_mask'] = False

        if 'amp' not in cfg:
            if cfg['dtype']=='amp':
                cfg['dtype'] = 'fp32'
                cfg['amp'] = True
            else:
                cfg['amp'] = False
        return cfg