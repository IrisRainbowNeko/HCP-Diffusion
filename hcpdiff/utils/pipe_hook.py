from typing import Union, List, Optional, Callable, Dict, Any

import PIL
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from .inpaint_pipe import preprocess_mask, preprocess_image, StableDiffusionInpaintPipelineLegacy
from einops import repeat

class HookPipe_T2I(StableDiffusionPipeline):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device('cuda')

    @property
    def device(self) -> torch.device:
        return torch.device('cuda')

    def proc_prompt(self, device, num_inference_steps, prompt_embeds = None, negative_prompt_embeds = None) -> List[torch.Tensor]:
        if not isinstance(prompt_embeds, list): # to emb for each step
            prompt_embeds = [prompt_embeds]*num_inference_steps
        if not isinstance(negative_prompt_embeds, list): # to emb for each step
            negative_prompt_embeds = [negative_prompt_embeds]*num_inference_steps

        prompt_embeds = [p.to(dtype=self.text_encoder.dtype, device=device) for p in prompt_embeds]
        negative_prompt_embeds = [p.to(dtype=self.text_encoder.dtype, device=device) for p in negative_prompt_embeds]

        prompt_embeds = [torch.cat([emb_neg, emb_pos]) for emb_pos, emb_neg in zip(prompt_embeds, negative_prompt_embeds)]
        return prompt_embeds # List[emb_step_i]*num_inference_steps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        pooled_output: Optional[torch.FloatTensor] = None,
        crop_coord: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size*self.vae_scale_factor
        width = width or self.unet.config.sample_size*self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt_embeds, list):
            batch_size = prompt_embeds[0].shape[0]
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale>1.0

        # 3. Encode input prompt
        prompt_embeds = self.proc_prompt(device, num_inference_steps,
                            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # SDXL inputs
        if pooled_output is not None:
            if crop_coord is None:
                crop_info = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float)
            else:
                crop_info = torch.tensor([height, width, *crop_coord], dtype=torch.float)
            crop_info = crop_info.to(device).repeat(batch_size, 1)
            pooled_output = pooled_output.to(device)

            if do_classifier_free_guidance:
                crop_info = torch.cat([crop_info, crop_info], dim=0)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps)-num_inference_steps*self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if pooled_output is None:
                    noise_pred = self.unet(latent_model_input, t, prompt_embeds[i], encoder_attention_mask=encoder_attention_mask,
                                           cross_attention_kwargs=cross_attention_kwargs, ).sample
                else:
                    added_cond_kwargs = {"text_embeds":pooled_output, "time_ids":crop_info}
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, prompt_embeds[i], encoder_attention_mask=encoder_attention_mask,
                                           cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

                # x_t -> x_0
                alpha_prod_t = alphas_cumprod[t.long()]
                beta_prod_t = 1-alpha_prod_t
                latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0

                # compute the previous noisy sample x_t -> x_t-1
                sc_out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = sc_out.prev_sample

                # call the callback, if provided
                if i == len(timesteps)-1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i%callback_steps == 0:
                        latents = callback(i, t, num_inference_steps, latents_x0, latents)
                        if latents is None:
                            return None

        latents = latents.to(dtype=self.vae.dtype)
        if not output_type == "latent":
            image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

class HookPipe_I2I(StableDiffusionImg2ImgPipeline):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device('cuda')

    @property
    def device(self) -> torch.device:
        return torch.device('cuda')

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        pooled_output: Optional[torch.FloatTensor] = None,
        crop_coord: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale>1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)
        image = repeat(image, 'n ... -> (n b) ...', b=batch_size)
        height, width = image.shape[2:]

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size*num_images_per_prompt)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
        ).to(self.unet.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # SDXL inputs
        if pooled_output is not None:
            if crop_coord is None:
                crop_info = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float)
            else:
                crop_info = torch.tensor([height, width, *crop_coord], dtype=torch.float)
            crop_info = crop_info.to(device).repeat(batch_size*num_images_per_prompt, 1)
            pooled_output = pooled_output.to(device)

            if do_classifier_free_guidance:
                crop_info = torch.cat([crop_info, crop_info], dim=0)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps)-num_inference_steps*self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                if pooled_output is None:
                    noise_pred = self.unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                           cross_attention_kwargs=cross_attention_kwargs, ).sample
                else:
                    added_cond_kwargs = {"text_embeds":pooled_output, "time_ids":crop_info}
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask,
                                           cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

                # x_t -> x_0
                alpha_prod_t = alphas_cumprod[t.long()]
                beta_prod_t = 1-alpha_prod_t
                latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps)-1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i%callback_steps == 0:
                        latents = callback(i, t, num_inference_steps, latents_x0, latents)
                        if latents is None:
                            return None

        latents = latents.to(dtype=self.vae.dtype)
        if not output_type == "latent":
            image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        has_nsfw_concept = None

        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

class HookPipe_Inpaint(StableDiffusionInpaintPipelineLegacy):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device('cuda')

    @property
    def device(self) -> torch.device:
        return torch.device('cuda')

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        add_predicted_noise: Optional[bool] = False,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        **kwargs
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale>1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image and mask
        if not isinstance(image, torch.FloatTensor):
            image = preprocess_image(image, batch_size).to(self._execution_device)

        mask_image = preprocess_mask(mask_image, batch_size, self.vae_scale_factor)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size*num_images_per_prompt)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)

        # 6. Prepare latent variables
        # encode the init image into latents and scale the latents
        latents, init_latents_orig, noise = self.prepare_latents(
            image, latent_timestep, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )

        # 7. Prepare mask latent
        mask = mask_image.to(device=self._execution_device, dtype=latents.dtype)
        mask = torch.cat([mask]*num_images_per_prompt)

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps)-num_inference_steps*self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, prompt_embeds, encoder_attention_mask=encoder_attention_mask).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

                # masking
                if add_predicted_noise:
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_orig, noise_pred_uncond, torch.tensor([t])
                    )
                else:
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))

                # x_t-1 -> x_0
                alpha_prod_t = alphas_cumprod[t.long()]
                beta_prod_t = 1-alpha_prod_t
                latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0
                # normalize latents_x0 to keep the contrast consistent with the original image
                latents_x0 = (latents_x0-latents_x0.mean())/latents_x0.std()*init_latents_orig.std()+init_latents_orig.mean()
                latents_x0 = (init_latents_orig*mask)+(latents_x0*(1-mask))

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                latents = (init_latents_proper*mask)+(latents*(1-mask))

                # call the callback, if provided
                if i == len(timesteps)-1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i%callback_steps == 0:
                        latents = callback(i, t, num_inference_steps, latents_x0, latents)
                        if latents is None:
                            return None

        # use original latents corresponding to unmasked portions of the image
        latents = (init_latents_orig*mask)+(latents*(1-mask))
        latents = latents.to(dtype=self.vae.dtype)
        if not output_type == "latent":
            image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        has_nsfw_concept = None

        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

class HCPSDPipe(StableDiffusionImg2ImgPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor)

        self.mask_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False, resample='nearest')

    @property
    def _execution_device(self) -> torch.device:
        return torch.device('cuda')

    @property
    def device(self) -> torch.device:
        return torch.device('cuda')

    def preprocess_images(self, image, mask, batch_size, num_images_per_prompt):
        if image is not None:
            if not isinstance(image, torch.FloatTensor):
                image = self.image_processor.preprocess(image, batch_size)
            if image.shape[0] == 1:
                image = repeat(image, 'n ... -> (n b) ...', b=batch_size)
        if mask is not None:
            mask = preprocess_mask(mask, batch_size, self.vae_scale_factor)
            mask = mask.to(device=self._execution_device, dtype=self.unet.dtype)
            mask = torch.cat([mask]*num_images_per_prompt)
        return image, mask

    def process_timesteps(self, num_inference_steps, strength, batch_size, num_images_per_prompt):
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, self._execution_device)
        latent_timestep = timesteps[:1].repeat(batch_size*num_images_per_prompt)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
        return timesteps, num_inference_steps, latent_timestep, alphas_cumprod

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        add_predicted_noise: Optional[bool] = False,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        **kwargs
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale>1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image and mask
        image, mask = self.preprocess_images(image, mask_image, batch_size, num_images_per_prompt)

        # 5. set timesteps
        timesteps, num_inference_steps, latent_timestep, alphas_cumprod = self.process_timesteps(
            num_inference_steps, strength, batch_size, num_images_per_prompt)

        # 6. Prepare latent variables
        # encode the init image into latents and scale the latents
        latents, init_latents_orig, noise = self.prepare_latents(
            image, latent_timestep, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )

        # 7. Prepare mask latent
        mask = mask_image.to(device=self._execution_device, dtype=latents.dtype)
        mask = torch.cat([mask]*num_images_per_prompt)

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps)-num_inference_steps*self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents]*2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)

                # masking
                if add_predicted_noise:
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_orig, noise_pred_uncond, torch.tensor([t])
                    )
                else:
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))

                # x_t-1 -> x_0
                alpha_prod_t = alphas_cumprod[t.long()]
                beta_prod_t = 1-alpha_prod_t
                latents_x0 = (latents-beta_prod_t**(0.5)*noise_pred)/alpha_prod_t**(0.5)  # approximate x_0
                # normalize latents_x0 to keep the contrast consistent with the original image
                latents_x0 = (latents_x0-latents_x0.mean())/latents_x0.std()*init_latents_orig.std()+init_latents_orig.mean()
                latents_x0 = (init_latents_orig*mask)+(latents_x0*(1-mask))

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                latents = (init_latents_proper*mask)+(latents*(1-mask))

                # call the callback, if provided
                if i == len(timesteps)-1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i%callback_steps == 0:
                        if callback(i, t, num_inference_steps, latents_x0):
                            return None

        # use original latents corresponding to unmasked portions of the image
        latents = (init_latents_orig*mask)+(latents*(1-mask))

        if not output_type == "latent":
            image = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        has_nsfw_concept = None

        do_denormalize = [True]*image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
