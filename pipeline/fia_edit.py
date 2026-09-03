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


from misc.frequency_utils import freq_com
from pipeline.common import RFEditingFluxPipeline, RFEditingSD35Pipeline

class RFFIAEditFluxPipeline(RFEditingFluxPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        source_prompt_2: str | list[str] | None = None,
        target_prompt_2: str | list[str] | None = None,
        inject_step: int=6,
        noise_coeff: list[float] = [0.1, 0.1 ,0.8],
        scale_coeff: float = 0.3,
        offset_coeff: float = 0.8,
        num_inference_steps: int = 28,
        num_average_steps: int = 1,
        source_guidance_scale: float = 2,
        target_guidance_scale: float = 5.5,
        interpolate_start_step: int = 0,
        interpolate_end_step: int = 24,
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
        clear_memory: bool = True,
        generator: torch.Generator | None = None,
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

      image_seq_len = source_img_latents.shape[1]
      mu = calculate_shift(
          image_seq_len,
          self.scheduler.config.base_image_seq_len,
          self.scheduler.config.max_image_seq_len,
          self.scheduler.config.base_shift,
          self.scheduler.config.max_shift,
      )
      timesteps, _ = retrieve_timesteps(
          self.scheduler,
          num_inference_steps,
          device,
          mu=mu,
      )
      num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
      dtype = source_img_latents.dtype
      using_inject_list = [True] * inject_step + [False] * (len(timesteps) - inject_step) #去最后的inject_step步保存他们的value值，这让刚开始的去噪效果更好

      if self.transformer.config.guidance_embeds:
          source_guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
          target_guidance = torch.full([1], target_guidance_scale, device=device, dtype=torch.float32)
          source_guidance = source_guidance.expand(source_img_latents.shape[0])
          target_guidance = target_guidance.expand(source_img_latents.shape[0])
      else:
          source_guidance = None
          target_guidance = None

      source_img_latents_edit = source_img_latents.clone()
      with self.progress_bar(total=interpolate_end_step) as progress_bar:
          for step_idx, t_curr in enumerate(timesteps):
              if num_inference_steps - step_idx > interpolate_end_step:
                  continue
              sigma_curr = t_curr / self.scheduler.config.num_train_timesteps

              if num_inference_steps - step_idx > interpolate_start_step:
                  delta_avg = torch.zeros_like(source_img_latents)
                  for _ in range(num_average_steps):
                      forward_noise = torch.randn(
                          source_img_latents.shape,
                          dtype=dtype,
                          device=device,
                          layout=source_img_latents.layout,
                          generator=generator,
                      )
                      source_img_latents_noisy = (
                          1 - sigma_curr
                      ) * source_img_latents + sigma_curr * forward_noise
                      target_img_latents_noisy = (
                          source_img_latents_edit + source_img_latents_noisy - source_img_latents
                      )

                      joint_attention_kwargs["inverse"] = True #是否invert
                      joint_attention_kwargs["editing_strategy"] = "replace_qkv"
                      joint_attention_kwargs["fusion_strategy"] = "fusion_qk"
                      joint_attention_kwargs["inject"] = using_inject_list[step_idx] #应该是在denoise的时候才会用，表示是否要inject,或者保存要inject的内容根据inverse参数来确定
                      joint_attention_kwargs["timestep"] = int(t_curr.item()) #输入timestep
                      joint_attention_kwargs["noise_coeff"] = noise_coeff
                      joint_attention_kwargs["scale_coeff"] = scale_coeff
                      joint_attention_kwargs["offset_coeff"] = offset_coeff
                      
                      noise_pred_source = self.transformer(
                          hidden_states=source_img_latents_noisy,
                          timestep=sigma_curr.expand(source_img_latents.shape[0]).to(dtype),
                          guidance=source_guidance,
                          pooled_projections=source_pooled_prompt_embeds,
                          encoder_hidden_states=source_prompt_embeds,
                          txt_ids=source_text_ids,
                          img_ids=source_latent_image_ids,
                          joint_attention_kwargs=joint_attention_kwargs,
                          return_dict=False,
                      )[0].float()

                      joint_attention_kwargs["inverse"] = False #是否invert

                      noise_pred_target = self.transformer(
                          hidden_states=target_img_latents_noisy,
                          timestep=sigma_curr.expand(target_img_latents_noisy.shape[0]).to(dtype),
                          guidance=target_guidance,
                          pooled_projections=target_pooled_prompt_embeds,
                          encoder_hidden_states=target_prompt_embeds,
                          txt_ids=target_text_ids,
                          img_ids=source_latent_image_ids,
                          joint_attention_kwargs=joint_attention_kwargs,
                          return_dict=False,
                      )[0].float()

                      delta_avg += (1 / num_average_steps) * (noise_pred_target - noise_pred_source)

                  source_img_latents_edit = self.scheduler.step(
                      delta_avg, t_curr, source_img_latents_edit, return_dict=False
                  )[0]
                  source_img_latents_edit = source_img_latents_edit.to(dtype)
              else:
                  if step_idx == num_inference_steps - interpolate_start_step:
                      forward_noise = torch.randn(
                          source_img_latents.shape,
                          dtype=dtype,
                          device=device,
                          layout=source_img_latents.layout,
                          generator=generator,
                      )
                      source_img_latents_noisy = (
                          1 - sigma_curr
                      ) * source_img_latents + sigma_curr * forward_noise
                      target_img_latents_noisy = (
                          source_img_latents_edit + source_img_latents_noisy - source_img_latents
                      )

                  noise_pred_target = self.transformer(
                      hidden_states=target_img_latents_noisy,
                      timestep=sigma_curr.expand(target_img_latents_noisy.shape[0]).to(dtype),
                      guidance=target_guidance,
                      pooled_projections=target_pooled_prompt_embeds,
                      encoder_hidden_states=target_prompt_embeds,
                      txt_ids=target_text_ids,
                      img_ids=source_latent_image_ids,
                      joint_attention_kwargs=joint_attention_kwargs,
                      return_dict=False,
                  )[0].float()
                  target_img_latents_noisy = self.scheduler.step(
                      noise_pred_target, t_curr, target_img_latents_noisy, return_dict=False
                  )[0]
                  target_img_latents_noisy = target_img_latents_noisy.to(dtype)

              if callback_on_step_end is not None:
                  callback_kwargs = {}
                  local_vars = locals()
                  for k in callback_on_step_end_tensor_inputs:
                      if k not in local_vars:
                          continue
                      callback_kwargs[k] = local_vars[k]
                  callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

              # call the callback, if provided
              if step_idx == len(timesteps) - 2 or (
                  (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
              ):
                  progress_bar.update()
        
    
      latents = source_img_latents_edit if interpolate_start_step == 0 else target_img_latents_noisy
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

      


class RFFIAEditSD35Pipeline(RFEditingSD35Pipeline):
    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        negative_prompt: str | list[str],
        inject_step: int=6,
        noise_coeff: list[float] = [0.1, 0.1 ,0.8],
        scale_coeff: float = 0.3,
        offset_coeff: float = 0.8,
        num_inference_steps: int = 28,
        num_average_steps: int = 1,
        source_guidance_scale: float = 2,
        target_guidance_scale: float = 5.5,
        interpolate_start_step: int = 0,
        interpolate_end_step: int = 24,
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
        clear_memory: bool = True,
        generator: torch.Generator | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        self._guidance_scale = target_guidance_scale
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
            do_classifier_free_guidance=target_guidance_scale>1, # Usually no CFG for inversion
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
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
            do_classifier_free_guidance=target_guidance_scale > 1, # Check guidance_scale > 1
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0) 
        do_classifier_free_guidance = target_guidance_scale > 1.0
        dtype = source_img_latents.dtype
        using_inject_list = [True] * inject_step + [False] * (len(timesteps) - inject_step) #去最后的inject_step步保存他们的value值，这让刚开始的去噪效果更好

        source_img_latents_edit = source_img_latents.clone()
        with self.progress_bar(total=interpolate_end_step) as progress_bar:
            for step_idx, t_curr in enumerate(timesteps):
                if num_inference_steps - step_idx > interpolate_end_step:
                    continue
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps

                if num_inference_steps - step_idx > interpolate_start_step:
                    delta_avg = torch.zeros_like(source_img_latents)
                    for _ in range(num_average_steps):
                        forward_noise = torch.randn(
                            source_img_latents.shape,
                            dtype=dtype,
                            device=device,
                            layout=source_img_latents.layout,
                            generator=generator,
                        )
                        source_img_latents_noisy = (
                            1 - sigma_curr
                        ) * source_img_latents + sigma_curr * forward_noise
                        target_img_latents_noisy = (
                            source_img_latents_edit + source_img_latents_noisy - source_img_latents
                        )

                        joint_attention_kwargs["inverse"] = True #是否invert
                        joint_attention_kwargs["editing_strategy"] = "replace_qkv"
                        joint_attention_kwargs["fusion_strategy"] = "fusion_qk"
                        joint_attention_kwargs["inject"] = using_inject_list[step_idx] #应该是在denoise的时候才会用，表示是否要inject,或者保存要inject的内容根据inverse参数来确定
                        joint_attention_kwargs["timestep"] = int(t_curr.item()) #输入timestep
                        joint_attention_kwargs["noise_coeff"] = noise_coeff
                        joint_attention_kwargs["scale_coeff"] = scale_coeff
                        joint_attention_kwargs["offset_coeff"] = offset_coeff
                        
                        if do_classifier_free_guidance:
                            # Concatenate latents: [uncond, cond]
                            latent_model_input_source = torch.cat([source_img_latents_noisy] * 2)
                            # Concatenate prompts: [negative, positive]
                            prompt_embeds_input_source = torch.cat([source_negative_prompt_embeds, source_prompt_embeds])
                            pooled_prompt_embeds_input_source = torch.cat([source_negative_pooled_prompt_embeds, source_pooled_prompt_embeds])
                        else:
                            latent_model_input_source = latents
                            prompt_embeds_input_source = source_prompt_embeds
                            pooled_prompt_embeds_input_source = source_pooled_prompt_embeds
                        
                        noise_pred_source = self.transformer(
                            hidden_states=latent_model_input_source,
                            timestep=t_curr.expand(latent_model_input_source.shape[0]),
                            pooled_projections=pooled_prompt_embeds_input_source,
                            encoder_hidden_states=prompt_embeds_input_source,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]


                        joint_attention_kwargs["inverse"] = False #是否invert

                        if do_classifier_free_guidance:
                            # Concatenate latents: [uncond, cond]
                            latent_model_input_target = torch.cat([target_img_latents_noisy] * 2)
                            # Concatenate prompts: [negative, positive]
                            prompt_embeds_input_target = torch.cat([target_negative_prompt_embeds, target_prompt_embeds])
                            pooled_prompt_embeds_input_target = torch.cat([target_negative_pooled_prompt_embeds, target_pooled_prompt_embeds])
                        else:
                            latent_model_input_target = target_img_latents_noisy
                            prompt_embeds_input_target = target_prompt_embeds
                            pooled_prompt_embeds_input_target = target_pooled_prompt_embeds

                        noise_pred_target = self.transformer(
                            hidden_states=latent_model_input_target,
                            timestep=t_curr.expand(latent_model_input_target.shape[0]),
                            pooled_projections=pooled_prompt_embeds_input_target,
                            encoder_hidden_states=prompt_embeds_input_target,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond_source, noise_pred_text_source = noise_pred_source.chunk(2)
                            noise_pred_source = noise_pred_uncond_source + target_guidance_scale * (noise_pred_text_source - noise_pred_uncond_source)
                            noise_pred_uncond_target, noise_pred_text_target = noise_pred_target.chunk(2)
                            noise_pred_target = noise_pred_uncond_target + target_guidance_scale * (noise_pred_text_target - noise_pred_uncond_target)

                        delta_avg += (1 / num_average_steps) * (noise_pred_target - noise_pred_source)

                    source_img_latents_edit = self.scheduler.step(
                        delta_avg, t_curr, source_img_latents_edit, return_dict=False
                    )[0]
                    source_img_latents_edit = source_img_latents_edit.to(dtype)
                else:
                    if step_idx == num_inference_steps - interpolate_start_step:
                        forward_noise = torch.randn(
                            source_img_latents.shape,
                            dtype=dtype,
                            device=device,
                            layout=source_img_latents.layout,
                            generator=generator,
                        )
                        source_img_latents_noisy = (
                            1 - sigma_curr
                        ) * source_img_latents + sigma_curr * forward_noise
                        target_img_latents_noisy = (
                            source_img_latents_edit + source_img_latents_noisy - source_img_latents
                        )

                    if do_classifier_free_guidance:
                        # Concatenate latents: [uncond, cond]
                        latent_model_input_target = torch.cat([target_img_latents_noisy] * 2)
                        # Concatenate prompts: [negative, positive]
                        prompt_embeds_input_target = torch.cat([target_negative_prompt_embeds, target_prompt_embeds])
                        pooled_prompt_embeds_input_target = torch.cat([target_negative_pooled_prompt_embeds, target_pooled_prompt_embeds])
                    else:
                        latent_model_input_target = target_img_latents_noisy
                        prompt_embeds_input_target = target_prompt_embeds
                        pooled_prompt_embeds_input_target = target_pooled_prompt_embeds

                    noise_pred_target = self.transformer(
                        hidden_states=latent_model_input_target,
                        timestep=t_curr.expand(latent_model_input_target.shape[0]),
                        pooled_projections=pooled_prompt_embeds_input_target,
                        encoder_hidden_states=prompt_embeds_input_target,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    
                    if do_classifier_free_guidance:
                       noise_pred_uncond_target, noise_pred_text_target = noise_pred_target.chunk(2)
                       noise_pred_target = noise_pred_uncond_target + target_guidance_scale * (noise_pred_text_target - noise_pred_uncond_target)

                    target_img_latents_noisy = self.scheduler.step(
                        noise_pred_target, t_curr, target_img_latents_noisy, return_dict=False
                    )[0]
                    target_img_latents_noisy = target_img_latents_noisy.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                # call the callback, if provided
                if step_idx == len(timesteps) - 2 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
            
        
        latents = source_img_latents_edit if interpolate_start_step == 0 else target_img_latents_noisy

        
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
        inject_step: int = 4,
        noise_coeff: list[float] = [0.1, 0.1 ,0.8],
        scale_coeff: float = 0.3,
        smooth_coeff: float = 0.999,
        num_inference_steps: int = 8,
        guidance_scale: float = 2,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        clear_memory: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
        ):
        generate_images = []
        for target_prompt in prompt_sequence:
            image = self(
                source_img=source_img,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                inject_step=inject_step,
                noise_coeff=noise_coeff,
                scale_coeff=scale_coeff,
                smooth_coeff=smooth_coeff,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                output_type=output_type,
                return_dict=return_dict,
                clear_memory=clear_memory,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]
            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images