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
from processor.ft_editing_attn_processor import AdalayernormReplace, register_norm_control_flux, AdalayernormReplaceSD35, register_norm_control_sd35


class FTEditFluxPipeline(RFEditingFluxPipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        target_pooled_prompt_embeds: torch.Tensor,
        target_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: int,
        device: torch.device,
        source_pooled_prompt_embeds: torch.Tensor | None = None,
        source_prompt_embeds: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        fixed_point_steps: int = 3,
        skip_steps: int = 0,
        latents_list: list[torch.Tensor] | None = None,
        record_latents: bool = False,
        inverse: bool = False,
        do_residual: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if do_residual:
            latents = torch.cat([latents, latents], dim=0)
            prompt_embeds = torch.cat([source_prompt_embeds, target_prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [source_pooled_prompt_embeds, target_pooled_prompt_embeds], dim=0
            )
        else:
            prompt_embeds = target_prompt_embeds
            pooled_prompt_embeds = target_pooled_prompt_embeds

        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps + 1,
            device,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - 1 - num_inference_steps * self.scheduler.order, 0)
        if inverse:
            timesteps = torch.flip(timesteps, [0])

        dtype = latents.dtype

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        residual_list = []
        latent_list = [latents.clone()]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                joint_attention_kwargs["do_replace"] = not inverse
                avg_noise_pred = None

                for fixed_point_step_idx in range(fixed_point_steps):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0].float()

                    if fixed_point_step_idx == 0:
                        avg_noise_pred = noise_pred
                    else:
                        avg_noise_pred = (fixed_point_step_idx * avg_noise_pred + noise_pred) / (
                            fixed_point_step_idx + 1
                        )
                noise_pred = avg_noise_pred
                latents = latents + (sigma_prev - sigma_curr) * noise_pred

                if do_residual:
                    src_latents = latents[0].clone()
                    residual = latents_list[-2 - (step_idx - skip_steps)] - src_latents
                    residual_list.append(residual)
                    latents[0, :] += residual.squeeze(0)
                latents = latents.to(dtype)
                if record_latents:
                    latent_list.append(latents.clone())

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_prev, callback_kwargs)

                # call the callback, if provided
                if step_idx == len(timesteps) - 2 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if do_residual:
            latents = latents[1:2]

        return latents, latent_list, residual_list

    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        source_prompt_2: str | list[str] | None = None,
        target_prompt_2: str | list[str] | None = None,
        num_inference_steps: int = 28,
        fixed_point_steps: int = 3,
        skip_steps: int = 0,
        ly_ratio: float = 1.0,
        guidance_scale: float = 2,
        num_images_per_prompt: int | None = 1,
        latents: torch.FloatTensor | None = None,
        source_prompt_embeds: torch.FloatTensor | None = None,
        source_pooled_prompt_embeds: torch.FloatTensor | None = None,
        target_prompt_embeds: torch.FloatTensor | None = None,
        target_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        joint_attention_kwargs: dict[str, Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if not self.initialize_processor:
            raise ValueError("Please call `add_processor` before running the pipeline.")

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

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )
        inverse_latents, latent_list, _ = self.denoise(
            latents=source_img_latents,
            target_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            record_latents=True,
            fixed_point_steps=fixed_point_steps,
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

        controller = AdalayernormReplace(
            prompts=[source_prompt, target_prompt],
            num_steps=num_inference_steps,
            ly_ratio=ly_ratio,
            tokenizer=self.tokenizer_2,
            device=device,
            num_adanorm=19,
        )
        register_norm_control_flux(self, controller)

        latents, *_ = self.denoise(
            latents=inverse_latents,
            latents_list=latent_list,
            source_pooled_prompt_embeds=source_pooled_prompt_embeds,
            source_prompt_embeds=source_prompt_embeds,
            target_pooled_prompt_embeds=target_pooled_prompt_embeds,
            target_prompt_embeds=target_prompt_embeds,
            text_ids=target_text_ids,
            latent_image_ids=source_latent_image_ids,  # reuse the same image ids
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            do_residual=True,
            skip_steps=skip_steps,
            fixed_point_steps=1,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        register_norm_control_flux(self, None)
        for processor in self.processors.values():
            if hasattr(processor, "controller"):
                processor.controller.cur_step = 0
                processor.controller.cur_layer = 0
        del latent_list

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            if target_size is not None:
                image = [
                    self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image
                ]

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
        guidance_scale: float = 2,
        num_inference_steps: int = 28,
        fixed_point_steps: int = 3,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
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

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )
        inverse_latents, *_ = self.denoise(
            latents=source_img_latents,
            target_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            fixed_point_steps=fixed_point_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents, *_ = self.denoise(
            latents=inverse_latents,
            target_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,  # reuse the same image ids
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            fixed_point_steps=fixed_point_steps,
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
            if target_size is not None:
                image = [
                    self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image
                ]

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
        guidance_scale: float = 2,
        fixed_point_steps: int = 3,
        skip_steps: int = 0,
        ly_ratio: float = 1.0,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
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
                fixed_point_steps=fixed_point_steps,
                skip_steps=skip_steps,
                ly_ratio=ly_ratio,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]

            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images


class FTEditSD35Pipeline(RFEditingSD35Pipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        target_pooled_prompt_embeds: torch.Tensor,
        target_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: int,
        device: torch.device,
        source_pooled_prompt_embeds: torch.Tensor | None = None,
        source_prompt_embeds: torch.Tensor | None = None,
        source_negative_prompt_embeds: torch.Tensor | None = None, # Added
        source_negative_pooled_prompt_embeds: torch.Tensor | None = None, # Added
        target_negative_prompt_embeds: torch.Tensor | None = None, # Added
        target_negative_pooled_prompt_embeds: torch.Tensor | None = None, # Added
        joint_attention_kwargs: dict[str, Any] | None = None,
        fixed_point_steps: int = 3,
        skip_steps: int = 0,
        latents_list: list[torch.Tensor] | None = None,
        record_latents: bool = False,
        inverse: bool = False,
        do_residual: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if do_residual: #source concat在前面，所以后面注意力替换的时候要用第一个替换后面的注意力
            latents = torch.cat([latents, latents], dim=0)
            prompt_embeds = torch.cat([source_prompt_embeds, target_prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [source_pooled_prompt_embeds, target_pooled_prompt_embeds], dim=0
            )
        else:
            prompt_embeds = target_prompt_embeds
            pooled_prompt_embeds = target_pooled_prompt_embeds

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        if inverse:
            timesteps = torch.flip(timesteps, [0])

        dtype = latents.dtype
        do_classifier_free_guidance = guidance_scale > 1.0 and not inverse

        residual_list = []
        latent_list = [latents.clone()]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                joint_attention_kwargs["do_replace"] = not inverse
                avg_noise_pred = None

                # --- Prepare Model Inputs for CFG ---
                if do_classifier_free_guidance:
                    # Concatenate latents: [uncond, cond_source, cond_target]
                    latent_model_input = torch.cat([latents] * 2)
                    # Concatenate prompts: [negative_source, negative_target, positive_source, positive_target]
                    prompt_embeds_input = torch.cat([source_negative_prompt_embeds, target_negative_prompt_embeds, prompt_embeds])
                    pooled_prompt_embeds_input = torch.cat([source_negative_pooled_prompt_embeds, target_negative_pooled_prompt_embeds,  pooled_prompt_embeds])
                else:
                    latent_model_input = latents
                    prompt_embeds_input = prompt_embeds
                    pooled_prompt_embeds_input = pooled_prompt_embeds

                for fixed_point_step_idx in range(fixed_point_steps):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_curr.expand(latent_model_input.shape[0]),
                        pooled_projections=pooled_prompt_embeds_input,
                        encoder_hidden_states=prompt_embeds_input,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond_source, noise_pred_uncond_target, noise_pred_text_source, noise_pred_text_target = noise_pred.chunk(4)
                        noise_pred_source = noise_pred_uncond_source + guidance_scale * (noise_pred_text_source - noise_pred_uncond_source)
                        noise_pred_target = noise_pred_uncond_target + guidance_scale * (noise_pred_text_target - noise_pred_uncond_target)
                        noise_pred = torch.cat([noise_pred_source, noise_pred_target], dim=0)

                    if fixed_point_step_idx == 0:
                        avg_noise_pred = noise_pred
                    else:
                        avg_noise_pred = (fixed_point_step_idx * avg_noise_pred + noise_pred) / (
                            fixed_point_step_idx + 1
                        )
                noise_pred = avg_noise_pred
                latents = latents + (sigma_prev - sigma_curr) * noise_pred

                if do_residual:
                    src_latents = latents[0].clone()
                    residual = latents_list[-2 - (step_idx - skip_steps)] - src_latents
                    residual_list.append(residual)
                    latents[0, :] += residual.squeeze(0)
                latents = latents.to(dtype)
                if record_latents:
                    latent_list.append(latents.clone())

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_prev, callback_kwargs)

                # call the callback, if provided
                if step_idx == len(timesteps) - 2 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if do_residual:
            latents = latents[1:2]

        return latents, latent_list, residual_list

    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        negative_prompt: str | list[str],
        num_inference_steps: int = 28,
        fixed_point_steps: int = 3,
        skip_steps: int = 0,
        ly_ratio: float = 1.0,
        guidance_scale: float = 2,
        num_images_per_prompt: int | None = 1,
        latents: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 256,
        joint_attention_kwargs: dict[str, Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        if not self.initialize_processor:
            raise ValueError("Please call `add_processor` before running the pipeline.")

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
            do_classifier_free_guidance=guidance_scale>1, # Usually no CFG for inversion
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )

        inverse_latents, latent_list, _ = self.denoise(
            latents=source_img_latents,
            target_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_prompt_embeds=source_prompt_embeds,
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            source_pooled_prompt_embeds=source_pooled_prompt_embeds,
            source_prompt_embeds=source_prompt_embeds,
            inverse=True,
            record_latents=True,
            fixed_point_steps=fixed_point_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        ) #inversion的时候不用residual

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
            do_classifier_free_guidance=guidance_scale > 1, # Check guidance_scale > 1
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        controller = AdalayernormReplaceSD35(
            prompts=[source_prompt, target_prompt] if guidance_scale == 1 else [negative_prompt, negative_prompt, source_prompt, target_prompt],
            num_steps=num_inference_steps,
            self_replace_steps=ly_ratio,
            tokenizer_clip=self.tokenizer,
            tokenizer_t5=self.tokenizer_3,
            device=device,
        )
        register_norm_control_sd35(self, controller)

        latents, *_ = self.denoise(
            latents=inverse_latents,
            latents_list=latent_list,
            source_pooled_prompt_embeds=source_pooled_prompt_embeds,
            source_prompt_embeds=source_prompt_embeds,
            target_pooled_prompt_embeds=target_pooled_prompt_embeds,
            target_prompt_embeds=target_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            source_negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            source_negative_prompt_embeds=source_negative_prompt_embeds,
            target_negative_pooled_prompt_embeds=target_negative_pooled_prompt_embeds,
            target_negative_prompt_embeds=target_negative_prompt_embeds,
            inverse=False,
            do_residual=True,
            skip_steps=skip_steps,
            fixed_point_steps=1,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        ) #这里需要residual

        register_norm_control_sd35(self, None)
        for processor in self.processors.values():
            if hasattr(processor, "controller"):
                processor.controller.cur_step = 0
                processor.controller.cur_layer = 0
        del latent_list

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            if target_size is not None:
                image = [
                    self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image
                ]

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
        source_prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt: str | list[str] = "",
        guidance_scale: float = 2,
        num_inference_steps: int = 28,
        fixed_point_steps: int = 3,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 256,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
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
            do_classifier_free_guidance=guidance_scale > 1.0, # Usually no CFG for inversion
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )


        inverse_latents, *_ = self.denoise(
            latents=source_img_latents,
            target_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_prompt_embeds=source_prompt_embeds,
            source_negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            source_negative_prompt_embeds=source_negative_prompt_embeds,
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            fixed_point_steps=fixed_point_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents, *_ = self.denoise(
            latents=inverse_latents,
            target_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_prompt_embeds=source_prompt_embeds,
            source_negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            source_negative_prompt_embeds=source_negative_prompt_embeds,
            target_negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            target_negative_prompt_embeds=source_negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            fixed_point_steps=fixed_point_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            if target_size is not None:
                image = [
                    self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image
                ]

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
        num_inference_steps: int = 25,
        guidance_scale: float = 2,
        fixed_point_steps: int = 3,
        skip_steps: int = 0,
        ly_ratio: float = 1.0,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 256,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
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
                fixed_point_steps=fixed_point_steps,
                skip_steps=skip_steps,
                ly_ratio=ly_ratio,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]

            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images
    
