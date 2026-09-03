from typing import Any, Callable

from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image

from pipeline.common import RFEditingFluxPipeline, RFEditingSD35Pipeline


class SMART_EditRFSolverEditFluxPipeline(RFEditingFluxPipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        inject_step: int,
        num_inference_steps: int,
        device: torch.device,
        joint_attention_kwargs: dict[str, Any] | None = None,
        token_indices: int | list[int] = 2,
        with_second_order: bool = False,
        inverse: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
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
        using_inject_list = [True] * inject_step + [False] * (len(timesteps[:-1]) - inject_step)
        if inverse:
            timesteps = torch.flip(timesteps, [0])
            using_inject_list = using_inject_list[::-1]

        dtype = latents.dtype

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                joint_attention_kwargs["inverse"] = inverse
                joint_attention_kwargs["second_order"] = False
                joint_attention_kwargs["inject"] = using_inject_list[step_idx]
                joint_attention_kwargs["timestep"] = int(t_prev.item()) if inverse else int(t_curr.item())
                
                # Mask mechanism from smart_edit_vanilla_edit.py
                joint_attention_kwargs["attention_weight_control"] = True
                joint_attention_kwargs["mask_fusion_strategy"] = "fusion_v"
                joint_attention_kwargs["token_indices"] = token_indices

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

                if not with_second_order:  # 单步ODE
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                    latents = latents.to(dtype)
                else:  # 两步ODE
                    mid_sample = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
                    mid_sample = mid_sample.to(dtype)

                    sigma_mid = torch.full(
                        (mid_sample.shape[0],),
                        (sigma_curr + (sigma_prev - sigma_curr) / 2),
                        dtype=mid_sample.dtype,
                        device=mid_sample.device,
                    )
                    joint_attention_kwargs["second_order"] = True
                    mid_noise_pred = self.transformer(
                        hidden_states=mid_sample,
                        timestep=sigma_mid.expand(latents.shape[0]).to(latents.dtype),
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0].float()

                    first_order = (mid_noise_pred - noise_pred) / ((sigma_prev - sigma_curr) / 2)
                    latents = (
                        latents
                        + (sigma_prev - sigma_curr) * noise_pred
                        + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order
                    )
                    latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                if step_idx == len(timesteps) - 2 or (
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
        inject_step: int = 6,
        token_indices: int | list[int] = 2,
        mask_prompt: str | None = None,
        num_inference_steps: int = 28,
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
        negative_prompt: str | list[str] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        with_second_order: bool = True,
        clear_memory: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if not self.initialize_processor:
            raise ValueError("Please call `add_processor` before running the pipeline.")

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        # Resolve token_indices from mask_prompt if provided (Flux: use tokenizer_2)
        if mask_prompt is not None and (token_indices == 2 or token_indices is None):
            try:
                from module_utils.vis_utils import get_token_indices
                token_indices = get_token_indices(self.tokenizer_2, source_prompt, mask_prompt)
            except Exception:
                token_indices = 2
        if isinstance(token_indices, int):
            token_indices = [token_indices]
        
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
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            token_indices=token_indices,
            inverse=True,
            with_second_order=with_second_order,
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
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            text_ids=target_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=guidance_scale,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            token_indices=token_indices,
            inverse=False,
            with_second_order=with_second_order,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if clear_memory:
            for processor in self.processors.values():
                if hasattr(processor, "clear_memory"):
                    processor.clear_memory()

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

class SMART_EditRFSolverEditSD35Pipeline(RFEditingSD35Pipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        inject_step: int,
        num_inference_steps: int,
        device: torch.device,
        joint_attention_kwargs: dict[str, Any] | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        token_indices_clip: int | list[int] = 2,
        token_indices_t5: int | list[int] = 2,
        with_second_order: bool = False,
        inverse: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        using_inject_list = [True] * inject_step + [False] * (len(timesteps[:-1]) - inject_step)
        
        if inverse:
            timesteps = torch.flip(timesteps, [0])
            using_inject_list = using_inject_list[::-1]

        dtype = latents.dtype
        do_classifier_free_guidance = guidance_scale > 1.0 and not inverse

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                joint_attention_kwargs["inverse"] = inverse
                joint_attention_kwargs["second_order"] = False
                joint_attention_kwargs["inject"] = using_inject_list[step_idx]
                joint_attention_kwargs["timestep"] = int(t_prev.item()) if inverse else int(t_curr.item())

                # Mask mechanism from val_vanilla_edit.py
                mask_container_hook = {}
                joint_attention_kwargs["attention_weight_control"] = True
                joint_attention_kwargs["mask_fusion_strategy"] = "fusion_v"
                joint_attention_kwargs["token_indices_clip"] = token_indices_clip
                joint_attention_kwargs["token_indices_t5"] = token_indices_t5
                joint_attention_kwargs["mask_hook"] = mask_container_hook

                # --- Prepare Model Inputs for CFG ---
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
                    pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
                else:
                    latent_model_input = latents
                    prompt_embeds_input = prompt_embeds
                    pooled_prompt_embeds_input = pooled_prompt_embeds

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_curr.expand(latent_model_input.shape[0]),
                    pooled_projections=pooled_prompt_embeds_input,
                    encoder_hidden_states=prompt_embeds_input,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if not with_second_order:  # 单步ODE
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                else:  # 两步ODE
                    mid_sample = latents + (sigma_prev - sigma_curr) / 2 * noise_pred

                    if do_classifier_free_guidance:
                        mid_latent_model_input = torch.cat([mid_sample] * 2)
                    else:
                        mid_latent_model_input = mid_sample

                    t_mid = torch.full(
                        (mid_latent_model_input.shape[0],),
                        (t_curr + (t_prev - t_curr) / 2),
                        dtype=t_curr.dtype,
                        device=t_curr.device,
                    )
                    joint_attention_kwargs["second_order"] = True
                    mid_noise_pred = self.transformer(
                        hidden_states=mid_latent_model_input,
                        timestep=t_mid.expand(mid_latent_model_input.shape[0]),
                        pooled_projections=pooled_prompt_embeds_input,
                        encoder_hidden_states=prompt_embeds_input,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        mid_noise_pred_uncond, mid_noise_pred_text = mid_noise_pred.chunk(2)
                        mid_noise_pred = mid_noise_pred_uncond + guidance_scale * (mid_noise_pred_text - mid_noise_pred_uncond)

                    first_order = (mid_noise_pred - noise_pred) / ((sigma_prev - sigma_curr) / 2)
                    latents = (
                        latents
                        + (sigma_prev - sigma_curr) * noise_pred
                        + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order
                    )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                if step_idx == len(timesteps) - 2 or (
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
        negative_prompt: str | list[str],
        mask_prompt: str | None = None,
        token_indices: int | list[int] | None = None,
        token_indices_clip: int | list[int] | None = None,
        token_indices_t5: int | list[int] | None = None,
        inject_step: int = 6,
        num_inference_steps: int = 28,
        guidance_scale: float = 2,
        num_images_per_prompt: int | None = 1,
        latents: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 256,
        joint_attention_kwargs: dict[str, Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        with_second_order: bool = True,
        clear_memory: bool = True,
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
        # Resolve token_indices from mask_prompt if provided
        if mask_prompt is not None and token_indices is None and token_indices_clip is None and token_indices_t5 is None:
            try:
                from module_utils.vis_utils import get_token_indices
                token_indices = get_token_indices(self.tokenizer_3, source_prompt, mask_prompt)
            except Exception:
                token_indices = 2
        if token_indices is not None:
            if isinstance(token_indices, int):
                token_indices = [token_indices]
            if token_indices_clip is None:
                token_indices_clip = token_indices
            if token_indices_t5 is None:
                token_indices_t5 = token_indices

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
            do_classifier_free_guidance=guidance_scale>1,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            guidance_scale=1,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            negative_prompt_embeds=source_negative_prompt_embeds,
            negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            token_indices_clip=token_indices_clip,
            token_indices_t5=token_indices_t5,
            inverse=True,
            with_second_order=with_second_order,
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
            do_classifier_free_guidance=guidance_scale > 1,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        latents = self.denoise(
            latents=inverse_latents,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            guidance_scale=guidance_scale,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            negative_prompt_embeds=target_negative_prompt_embeds,
            negative_pooled_prompt_embeds=target_negative_pooled_prompt_embeds,
            token_indices_clip=token_indices_clip,
            token_indices_t5=token_indices_t5,
            inverse=False,
            with_second_order=with_second_order,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if clear_memory:
            for processor in self.processors.values():
                if hasattr(processor, "clear_memory"):
                    processor.clear_memory()

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
        negative_prompt: str | list[str],
        guidance_scale: float = 2,
        num_inference_steps: int = 28,
        with_second_order: bool = True,
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
            do_classifier_free_guidance=guidance_scale > 1.0,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            guidance_scale=1,
            inject_step=0,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            negative_prompt_embeds=source_negative_prompt_embeds,
            negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            inverse=True,
            with_second_order=with_second_order,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents = self.denoise(
            latents=inverse_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            guidance_scale=guidance_scale,
            inject_step=0,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            negative_prompt_embeds=source_negative_prompt_embeds,
            negative_pooled_prompt_embeds=source_negative_pooled_prompt_embeds,
            inverse=False,
            with_second_order=with_second_order,
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
        negative_prompt: str | list[str],
        prompt_sequence: list[str | list[str]],
        inject_step: int = 1,
        token_indices_clip: int | list[int] = 2,
        token_indices_t5: int | list[int] = 2,
        num_inference_steps: int = 8,
        guidance_scale: float = 2,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 256,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        with_second_order: bool = False,
        clear_memory: bool = True,
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
                inject_step=inject_step,
                token_indices_clip=token_indices_clip,
                token_indices_t5=token_indices_t5,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                output_type=output_type,
                return_dict=return_dict,
                with_second_order=with_second_order,
                clear_memory=clear_memory,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]
            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images