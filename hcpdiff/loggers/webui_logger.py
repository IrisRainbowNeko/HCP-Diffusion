from typing import Dict, Any

from loguru import logger

from .cli_logger import CLILogger

class WebUILogger(CLILogger):
    def _log(self, datas: Dict[str, Any], step: int = 0):
        logger.info('this progress steps:'+', '.join([f"{k} = {v['format'].format(*v['data'])}" for k, v in datas.items()]))
