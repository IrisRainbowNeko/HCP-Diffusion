from sfast.compilers.diffusion_pipeline_compiler import (compile_unet, CompilationConfig)
from rainbowneko.infer import BasicAction


class SFastCompileAction(BasicAction):

    @staticmethod
    def compile_model(unet):
        # compile model
        config = CompilationConfig.Default()
        config.enable_xformers = False
        try:
            import xformers
            config.enable_xformers = True
        except ImportError:
            print('xformers not installed, skip')
        # NOTE:
        # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
        # Disable Triton if you encounter this problem.
        try:
            import tritonx
            config.enable_triton = True
        except ImportError:
            print('Triton not installed, skip')
        config.enable_cuda_graph = True

        return compile_unet(unet, config)

    def forward(self, denoiser, **states):
        denoiser = self.compile_model(denoiser)
        return {'denoiser': denoiser}