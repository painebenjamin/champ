from __future__ import annotations
# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import os
import gc
import re
import math
import json
import torch
import inspect
import logging
import numpy as np

from typing import Callable, List, Optional, Union

from PIL import Image
from contextlib import nullcontext
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available, is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPVisionConfig
)
from champ.models.unet_2d_condition import UNet2DConditionModel
from champ.models.unet_3d_condition import UNet3DConditionModel
from champ.models.guidance_encoder import GuidanceEncoder
from champ.models.mutual_self_attention import ReferenceAttentionControl

from champ.utils.context_utils import get_context_scheduler
from champ.utils.pipeline_utils import get_tensor_interpolation_method
from champ.utils.ckpt_utils import iterate_state_dict

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

logger = logging.getLogger(__name__)

@dataclass
class CHAMPPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class CHAMPPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        guidance_encoder_depth: Optional[GuidanceEncoder],
        guidance_encoder_normal: Optional[GuidanceEncoder],
        guidance_encoder_semantic_map: Optional[GuidanceEncoder],
        guidance_encoder_dwpose: Optional[GuidanceEncoder],
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ]
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            guidance_encoder_depth=guidance_encoder_depth,
            guidance_encoder_normal=guidance_encoder_normal,
            guidance_encoder_semantic_map=guidance_encoder_semantic_map,
            guidance_encoder_dwpose=guidance_encoder_dwpose,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    @classmethod
    def from_single_file(
        cls,
        file_path_or_repository: str,
        filename: str="champ.safetensors",
        config_filename: str="config.json",
        variant: Optional[str]=None,
        subfolder: Optional[str]=None,
        device: Optional[Union[str, torch.device]]=None,
        torch_dtype: Optional[torch.dtype]=None,
        cache_dir: Optional[str]=None,
        use_depth_guidance: bool=True,
        use_normal_guidance: bool=True,
        use_semantic_map_guidance: bool=True,
        use_dwpose_guidance: bool=True,
    ) -> CHAMPPipeline:
        """
        Load a CHAMP pipeline from a single file.
        """
        if variant is not None:
            filename, ext = os.path.splitext(filename)
            filename = f"{filename}.{variant}{ext}"

        if device is None:
            device = "cpu"
        else:
            device = str(device)

        if os.path.isdir(file_path_or_repository):
            model_dir = file_path_or_repository
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)
            file_path = os.path.join(model_dir, filename)
            config_path = os.path.join(model_dir, config_filename)
        elif os.path.isfile(file_path_or_repository):
            file_path = file_path_or_repository
            if os.path.isfile(config_filename):
                config_path = config_filename
            else:
                config_path = os.path.join(os.path.dirname(file_path), config_filename)
                if not os.path.exists(config_path) and subfolder:
                    config_path = os.path.join(os.path.dirname(file_path), subfolder, config_filename)
        elif re.search(r"^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$", file_path_or_repository):
            file_path = hf_hub_download(
                file_path_or_repository,
                filename,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
            try:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                )
            except:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    cache_dir=cache_dir,
                )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File {config_path} not found.")

        with open(config_path, "r") as f:
            champ_config = json.load(f)

        # Create the scheduler
        scheduler = DDIMScheduler(**champ_config["scheduler"])

        # Create the models
        with (init_empty_weights() if is_accelerate_available() else nullcontext()):
            # UNets
            reference_unet = UNet2DConditionModel.from_config(champ_config["reference_unet"])
            denoising_unet = UNet3DConditionModel.from_config(champ_config["denoising_unet"])

            # VAE
            vae = AutoencoderKL.from_config(champ_config["vae"])

            # Image encoder
            image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(**champ_config["image_encoder"]))

            # Guidance encoders
            if use_depth_guidance:
                guidance_encoder_depth = GuidanceEncoder(**champ_config["guidance_encoder"])
            else:
                guidance_encoder_depth = None

            if use_normal_guidance:
                guidance_encoder_normal = GuidanceEncoder(**champ_config["guidance_encoder"])
            else:
                guidance_encoder_normal = None

            if use_semantic_map_guidance:
                guidance_encoder_semantic_map = GuidanceEncoder(**champ_config["guidance_encoder"])
            else:
                guidance_encoder_semantic_map = None

            if use_dwpose_guidance:
                guidance_encoder_dwpose = GuidanceEncoder(**champ_config["guidance_encoder"])
            else:
                guidance_encoder_dwpose = None

        # Load the weights
        logger.debug("Models created, loading weights...")
        state_dicts = {}
        for key, value in iterate_state_dict(file_path):
            try:
                module, _, key = key.partition(".")
                if is_accelerate_available():
                    if module == "reference_unet":
                        set_module_tensor_to_device(reference_unet, key, device=device, value=value)
                    elif module == "denoising_unet":
                        set_module_tensor_to_device(denoising_unet, key, device=device, value=value)
                    elif module == "vae":
                        set_module_tensor_to_device(vae, key, device=device, value=value)
                    elif module == "image_encoder":
                        set_module_tensor_to_device(image_encoder, key, device=device, value=value)
                    elif module == "guidance_encoder_depth" and guidance_encoder_depth is not None:
                        set_module_tensor_to_device(guidance_encoder_depth, key, device=device, value=value)
                    elif module == "guidance_encoder_normal" and guidance_encoder_normal is not None:
                        set_module_tensor_to_device(guidance_encoder_normal, key, device=device, value=value)
                    elif module == "guidance_encoder_semantic_map" and guidance_encoder_semantic_map is not None:
                        set_module_tensor_to_device(guidance_encoder_semantic_map, key, device=device, value=value)
                    elif module == "guidance_encoder_dwpose" and guidance_encoder_dwpose is not None:
                        set_module_tensor_to_device(guidance_encoder_dwpose, key, device=device, value=value)
                    else:
                        raise ValueError(f"Unknown module: {module}")
                else:
                    if module not in state_dicts:
                        state_dicts[module] = {}
                    state_dicts[module][key] = value
            except (AttributeError, KeyError, ValueError) as ex:
                logger.warning(f"Skipping module {module} key {key} due to {type(ex)}: {ex}")
        if not is_accelerate_available():
            try:
                reference_unet.load_state_dict(state_dicts["reference_unet"])
                denoising_unet.load_state_dict(state_dicts["denoising_unet"])
                vae.load_state_dict(state_dicts["vae"])
                image_encoder.load_state_dict(state_dicts["image_encoder"], strict=False)
                if guidance_encoder_depth is not None:
                    guidance_encoder_depth.load_state_dict(state_dicts["guidance_encoder_depth"])
                if guidance_encoder_normal is not None:
                    guidance_encoder_normal.load_state_dict(state_dicts["guidance_encoder_normal"])
                if guidance_encoder_semantic_map is not None:
                    guidance_encoder_semantic_map.load_state_dict(state_dicts["guidance_encoder_semantic_map"])
                if guidance_encoder_dwpose is not None:
                    guidance_encoder_dwpose.load_state_dict(state_dicts["guidance_encoder_dwpose"])
                del state_dicts
                gc.collect()
            except KeyError as ex:
                raise RuntimeError(f"File did not provide a state dict for {ex}")

        # Create the pipeline
        pipeline = cls(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            guidance_encoder_depth=guidance_encoder_depth,
            guidance_encoder_normal=guidance_encoder_normal,
            guidance_encoder_semantic_map=guidance_encoder_semantic_map,
            guidance_encoder_dwpose=guidance_encoder_dwpose,
            scheduler=scheduler,
        )

        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        pipeline.to(device)

        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("XFormers is not available, falling back to PyTorch attention")
        return pipeline

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.denoising_unet, self.reference_unet, self.vae, self.image_encoder, self.guidance_encoder_depth, self.guidance_encoder_normal, self.guidance_encoder_semantic_map, self.guidance_encoder_dwpose]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    def images_from_video(
        self,
        video: torch.Tensor,
        rescale: bool=False
    ) -> List[Image.Image]:
        """
        Convert a video tensor to a list of PIL images
        """
        import numpy as np
        import torchvision
        from einops import rearrange
        video = rearrange(video, "b c t h w -> t b c h w")
        height, width = video.shape[-2:]
        outputs = []

        for x in video:
            x = torchvision.utils.make_grid(x, nrow=1)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            x = Image.fromarray(x)
            outputs.append(x)

        return outputs

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        multi_guidance_group,
        # guidance_types,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str]="pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare a list of pose condition images
        # guidance_lst = []
        # for multi_guidance_frame in zip(*(multi_guidance_group.values())):
        #     guidance_frame_lst = [torch.from_numpy(np.array(guidance_image.resize((width, height)))) / 255.0 for guidance_image in multi_guidance_frame]
        #     guidance_frame_tensor = torch.cat(guidance_frame_lst, dim=-1)  # (h, w, n_cond*c)
        #     guidance_frame_tensor = guidance_frame_tensor.permute(2, 0, 1).unsqueeze(1)  # （n_cond*c, 1, h, w)
        #     guidance_lst += [guidance_frame_tensor]

        # guidance_tensor = torch.cat(guidance_lst, dim=1).unsqueeze(0)
        # guidance_tensor = guidance_tensor.to(
        #     device=device, dtype=self.guidance_encoder_depth.dtype
        # )
        # guidance_tensor_group = torch.chunk(pose_cond_tensor, pose_cond_tensor.shape[1]//3, dim=1)
        # pose_fea_depth = self.guidance_encoder_depth(pose_cond_tensor_tuple[0])
        # pose_fea_normal = self.guidance_encoder_normal(pose_cond_tensor_tuple[1])
        # pose_fea_semantic_map = self.guidance_encoder_semantic_map(pose_cond_tensor_tuple[2])
        # pose_fea_dwpose = self.guidance_encoder_dwpose(pose_cond_tensor_tuple[3])
        
        guidance_fea_lst = []
        for guidance_type, guidance_pil_lst in multi_guidance_group.items():
            guidance_tensor_lst = [torch.from_numpy(np.array(guidance_image.resize((width, height)))) / 255.0 for guidance_image in guidance_pil_lst]
            guidance_tensor = torch.stack(guidance_tensor_lst, dim=0).permute(3, 0, 1, 2)  # (c, f, h, w)
            guidance_tensor = guidance_tensor.unsqueeze(0)  # (1, c, f, h, w)
            
            guidance_encoder = getattr(self, f"guidance_encoder_{guidance_type}")
            guidance_tensor = guidance_tensor.to(device, guidance_encoder.dtype)
            guidance_fea_lst += [guidance_encoder(guidance_tensor)]

        guidance_fea = torch.stack(guidance_fea_lst).sum(0)

        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        # t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        0,
                    )
                )
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = []
                for i in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            i * context_batch_size : (i + 1) * context_batch_size
                        ]
                    )

                for context in global_context:
                    # 3.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    b, c, f, h, w = latent_model_input.shape
                    latent_guidance_input = torch.cat(
                        [guidance_fea[:, :, c] for c in context]
                    ).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states[:b],
                        guidance_fea=latent_guidance_input,
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        if output_type == "tensor":
            images = torch.from_numpy(images)
        elif output_type == "pil":
            images = self.images_from_video(torch.from_numpy(images))

        if not return_dict:
            return images

        return CHAMPPipelineOutput(videos=images)
