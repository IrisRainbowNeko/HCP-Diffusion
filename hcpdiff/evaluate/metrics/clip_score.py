from torchmetrics.multimodal.clip_score import CLIPScore, _clip_score_update
from torch import Tensor
from typing import List

class CLIPScoreMetric(CLIPScore):
    def update(self, images: Tensor | List[Tensor], text: str | list[str]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors, in the [-1, 1] range
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        images = (images+1)/2 # [-1,1] -> [0,1]
        score, n_samples = _clip_score_update(images, text, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples
