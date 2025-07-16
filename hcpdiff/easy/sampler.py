from hcpdiff.diffusion.sampler import DiffusersSampler
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler

class Diffusers_SD:
    dpmpp_2m = DiffusersSampler(
        DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            algorithm_type='dpmsolver++',
        )
    )

    dpmpp_2m_karras = DiffusersSampler(
        DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            algorithm_type='dpmsolver++',
            use_karras_sigmas=True,
        )
    )

    ddim = DiffusersSampler(
        DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
        )
    )

    euler = DiffusersSampler(
        EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
        )
    )

    euler_a = DiffusersSampler(
        EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
        )
    )

    euler_flow = DiffusersSampler(
        FlowMatchEulerDiscreteScheduler(
            shift=3.0,
            use_dynamic_shifting=True,
        )
    )