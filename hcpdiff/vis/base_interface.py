
class BaseInterface:
    need_inter_imgs = False

    def __init__(self, show_steps=0):
        self.show_steps = show_steps

    def on_inter_step(self, i, num_steps, t, latents, images):
        pass

    def on_infer_finish(self, images, prompt, negative_prompt, save_cfg=False, seeds=None):
        pass
