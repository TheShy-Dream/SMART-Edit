from typing import Any, Callable

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image

from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from pipeline.common import RFEditingFluxPipeline, RFEditingSD35Pipeline


class RFInversionEditFluxPipeline(RFEditingFluxPipeline):
    """Modified from https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_flux_rf_inversion.py"""

    def get_timesteps(self, num_inference_steps, strength=1.0):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        sigmas = self.scheduler.sigmas[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, sigmas, num_inference_steps - t_start

    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        gamma: float,
        num_inference_steps: int,
        device: torch.device,
        original_latents: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        strength: float = 1.0,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        inverse: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        image_seq_len = latents.shape[1]
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - 1 - num_inference_steps * self.scheduler.order, 0)
        if inverse:
            inversion_aware_sigmas = torch.flip(self.scheduler.sigmas, [0])
            timesteps = torch.flip(timesteps, [0])
            gamma = [gamma] * len(timesteps)
        else:
            inversion_aware_sigmas = self.scheduler.sigmas
            gamma_steps = int(stop_timestep * len(timesteps))
            gamma = [gamma] * gamma_steps + [0] * (len(timesteps) - gamma_steps)
        gamma[:start_timestep] = [0] * start_timestep

        dtype = latents.dtype

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if inverse:
            random_noise = torch.randn_like(latents)

        timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, t_curr in enumerate(timesteps):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_norm = torch.tensor(step_idx / num_inference_steps, device=device)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=(sigma_norm if inverse else sigma_curr).expand(latents.shape[0]).to(dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0].float()
                noise_pred = noise_pred if inverse else -noise_pred

                if inverse:
                    cond_noise_pred = (random_noise - latents) / (1 - sigma_norm)
                else:
                    cond_noise_pred = (original_latents - latents) / sigma_curr
                delta_t = torch.abs(sigmas[step_idx] - sigmas[step_idx + 1])
                cond_noise_pred = noise_pred + gamma[step_idx] * (cond_noise_pred - noise_pred)
                latents = latents + delta_t * cond_noise_pred
                latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(
                        self,
                        step_idx,
                        inversion_aware_sigmas[step_idx + 1] * self.scheduler.config.num_train_timesteps,
                        callback_kwargs,
                    )

                # call the callback, if provided
                if step_idx == len(timesteps) - 1 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return latents

    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        source_prompt_2: str | list[str] | None = None,
        target_prompt_2: str | list[str] | None = None,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        gamma: float = 0.9,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        latents: torch.FloatTensor | None = None,
        source_prompt_embeds: torch.FloatTensor | None = None,
        source_pooled_prompt_embeds: torch.FloatTensor | None = None,
        target_prompt_embeds: torch.FloatTensor | None = None,
        target_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        height: int | None = None,
        width: int | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        device = self._execution_device
        (
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            source_text_ids,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt_2,
            prompt_embeds=source_prompt_embeds,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=(width, height)
        )

        original_latents = source_img_latents.clone()
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            gamma=0.5,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        (
            target_prompt_embeds,
            target_pooled_prompt_embeds,
            target_text_ids,
        ) = self.encode_prompt(
            prompt=target_prompt,
            prompt_2=target_prompt_2,
            prompt_embeds=target_prompt_embeds,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        latents = self.denoise(
            latents=inverse_latents,
            original_latents=original_latents,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            text_ids=target_text_ids,
            latent_image_ids=source_latent_image_ids,  # reuse the same image ids
            guidance_scale=guidance_scale,
            gamma=gamma,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            stop_timestep=stop_timestep,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            image = [self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    @torch.inference_mode()
    def reconstruction(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        source_prompt_2: str | list[str] | None = None,
        source_prompt_embeds: torch.FloatTensor | None = None,
        source_pooled_prompt_embeds: torch.FloatTensor | None = None,
        guidance_scale: float = 1,
        strength: float = 1.0,
        num_inference_steps: int = 25,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        gamma: float = 0.9,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        device = self._execution_device
        (
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            source_text_ids,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt_2,
            prompt_embeds=source_prompt_embeds,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=(width, height)
        )

        original_latents = source_img_latents.clone()
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            gamma=0.5,
            strength=strength,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents = self.denoise(
            latents=inverse_latents,
            original_latents=original_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,  # reuse the same image ids
            guidance_scale=guidance_scale,
            gamma=gamma,
            strength=strength,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            start_timestep=start_timestep,
            stop_timestep=stop_timestep,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            image = [self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    def multiturn(
        self,
        source_img: np.ndarray | torch.Tensor,
        source_prompt: str | list[str],
        prompt_sequence: list[str | list[str]],
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        gamma: float = 0.9,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        generate_images = []
        for target_prompt in prompt_sequence:
            image = self(
                source_img,
                source_prompt,
                target_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                output_type=output_type,
                return_dict=return_dict,
                start_timestep=start_timestep,
                stop_timestep=stop_timestep,
                gamma=gamma,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]
            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images


class RFInversionEditSD35Pipeline(RFEditingSD35Pipeline):
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        gamma: float,
        num_inference_steps: int,
        device: torch.device,
        original_latents: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None, # Added
        negative_pooled_prompt_embeds: torch.Tensor | None = None, # Added
        joint_attention_kwargs: dict[str, Any] | None = None,
        strength: float = 1.0,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        inverse: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        if inverse:
            inversion_aware_sigmas = torch.flip(self.scheduler.sigmas, [0])
            timesteps = torch.flip(timesteps, [0])
            gamma = [gamma] * len(timesteps)
        else:
            inversion_aware_sigmas = self.scheduler.sigmas
            gamma_steps = int(stop_timestep * len(timesteps))
            gamma = [gamma] * gamma_steps + [0] * (len(timesteps) - gamma_steps)
        gamma[:start_timestep] = [0] * start_timestep

        dtype = latents.dtype

        # --- CFG Setup ---
        # We perform CFG if scale > 1 and we are NOT in the inversion phase
        do_classifier_free_guidance = guidance_scale > 1.0 and not inverse

        if inverse:
            random_noise = torch.randn_like(latents)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, t_curr in enumerate(timesteps):
                
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_norm = torch.tensor(step_idx / num_inference_steps, device=device)

                # --- Prepare Model Inputs for CFG ---
                if do_classifier_free_guidance:
                    # Concatenate latents: [uncond, cond]
                    latent_model_input = torch.cat([latents] * 2)
                    # Concatenate prompts: [negative, positive]
                    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
                    pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
                    
                    timestep_input = (sigma_norm*self.scheduler.config.num_train_timesteps if inverse else t_curr).expand(latent_model_input.shape[0])#.to(dtype)
                else:
                    latent_model_input = latents
                    prompt_embeds_input = prompt_embeds
                    pooled_prompt_embeds_input = pooled_prompt_embeds
                    timestep_input = (sigma_norm*self.scheduler.config.num_train_timesteps if inverse else t_curr).expand(latents.shape[0])#.to(dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    pooled_projections=pooled_prompt_embeds_input, 
                    encoder_hidden_states=prompt_embeds_input,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0].float()

                # --- Perform CFG ---
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise_pred = noise_pred if inverse else -noise_pred 

                if inverse:
                    cond_noise_pred = (random_noise - latents) / (1 - sigma_norm) 
                else:
                    # u(X|y_0) - This is the "Structure Guidance" from the source image
                    cond_noise_pred = (original_latents - latents) / sigma_curr 
                
                delta_t = torch.abs(inversion_aware_sigmas[step_idx] - inversion_aware_sigmas[step_idx + 1])
                
                # Combine the Text Direction (noise_pred, potentially CFG-guided) 
                # with the Source Structure Direction (cond_noise_pred)
                cond_noise_pred = noise_pred + gamma[step_idx] * (cond_noise_pred - noise_pred)
                
                latents = latents + delta_t * cond_noise_pred 
                latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(
                        self,
                        step_idx,
                        inversion_aware_sigmas[step_idx + 1] * self.scheduler.config.num_train_timesteps,
                        callback_kwargs,
                    )

                if step_idx == len(timesteps) - 1 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return latents
    

    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        gamma: float = 0.9,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        latents: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 256,
        height: int | None = None,
        width: int | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}
    
    
        device = self._execution_device
        (
            source_prompt_embeds,
            source_negative_prompt_embeds,
            source_pooled_prompt_embeds,
            source_negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0, # Check guidance_scale > 1, # Usually no CFG for inversion
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )
        original_latents = source_img_latents.clone()
        
        # 2. Run Inversion (No CFG, guidance_scale=1)
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            guidance_scale=1.0, # Force 1.0 for inversion
            gamma=0.5,
            num_inference_steps=num_inference_steps,
            device=device,
            negative_prompt_embeds=source_negative_prompt_embeds,
            negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds, # Pass negative, but not use
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        (
            target_prompt_embeds,
            target_negative_prompt_embeds,
            target_pooled_prompt_embeds,
            target_negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=target_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=guidance_scale > 1.0, # Check guidance_scale > 1
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        latents = self.denoise(
            latents=inverse_latents, 
            original_latents=original_latents,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            negative_prompt_embeds=target_negative_prompt_embeds, # Pass negative
            negative_pooled_prompt_embeds=target_negative_pooled_prompt_embeds, # Pass negative
            guidance_scale=guidance_scale,
            gamma=gamma,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            stop_timestep=stop_timestep,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            image = [self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

    @torch.inference_mode()
    def reconstruction(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        guidance_scale: float = 3.5, # SD3.5 常用 CFG
        num_inference_steps: int = 28,
        strength: float = 1.0,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        gamma: float = 0.9,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 256,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        device = self._execution_device
        
        # Reconstruction 中 source 和 target 是同一个 prompt
        # 我们需要同时获取用于 Inversion (通常无 CFG) 和 Generation (可能有 CFG) 的 embedding
        (
            source_prompt_embeds,
            source_negative_prompt_embeds,
            source_pooled_prompt_embeds,
            source_negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )
        original_latents = source_img_latents.clone()

        # 注意：Inversion 通常不使用 CFG (guidance_scale=1.0)，只跟随 prompt 
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=None, # Inversion 一般不需要负面提示词
            negative_pooled_prompt_embeds=None,
            guidance_scale=1.0, 
            gamma=0.5, # Inversion 固定的 gamma
            strength=strength,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True, # 开启反向
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # 使用相同的 Prompt 和反向出来的 Latent 进行重建
        latents = self.denoise(
            latents=inverse_latents,
            original_latents=original_latents, # 传入原图 latents 用于结构引导 (Structure Guidance)
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=source_negative_prompt_embeds, # 传入负面 embedding 以支持 CFG
            negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            gamma=gamma,
            strength=strength,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False, # 正向生成
            start_timestep=start_timestep,
            stop_timestep=stop_timestep,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            # SD3 这里的 resize 逻辑通常不像 Flux 那样必须 pack/unpack，但为了保持尺寸一致性可以加上
            image = [self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image] 

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
    
    def multiturn(
        self,
        source_img: np.ndarray | torch.Tensor,
        source_prompt: str | list[str],
        prompt_sequence: list[str | list[str]],
        negative_prompt: str | list[str],
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 256,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        start_timestep: int = 0,
        stop_timestep: float = 0.25,
        gamma: float = 0.9,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        generate_images = []
        for target_prompt in prompt_sequence:
            image = self(
                source_img,
                source_prompt,
                target_prompt,
                negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                output_type=output_type,
                return_dict=return_dict,
                start_timestep=start_timestep,
                stop_timestep=stop_timestep,
                gamma=gamma,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]
            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images






        
    

