# https://huggingface.co/docs/diffusers/training/lora
import functools
import gc
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict
import accelerate
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.io import ImageReadMode
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, \
    DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.io import read_image, ImageReadMode
check_min_version("0.22.0.dev0")
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
save_dir = './weights/'
logs_dir = './lightning_logs/'

logger = get_logger(__name__)
if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False


def save_model_card(
        repo_id: str,
        images: list = None,
        validation_prompt: str = None,
        base_model: str = None,
        dataset_name: str = None,
        repo_folder: str = None,
        vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
{img_str}

Special VAE used for training: {vae_path}.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers-training",
        "diffusers",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            # aici 1
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    # There might have slightly performance improvement
    # by changing model_input.cpu() to accelerator.gather(model_input)
    return {"model_input": model_input.cpu()}


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def main(args):
    logging_dir = Path(args["output_dir"], args["logging_dir"])
    df1 = pd.read_csv('./newData/NewInput.csv')
    Samplesize = 30  # number of samples that you want
    df1.groupby(['category']).size()


    accelerator_project_config = ProjectConfiguration(project_dir=args["output_dir"], logging_dir=args["logging_dir"])

    if torch.backends.mps.is_available() and args["mixed_precision"] == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        mixed_precision=args["mixed_precision"],
        log_with="all",
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args["output_dir"] is not None:
            os.makedirs(args["output_dir"], exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args["pretrained_model_name_or_path"],
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args["pretrained_model_name_or_path"],
        subfolder="tokenizer_2",
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args["pretrained_model_name_or_path"]
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args["pretrained_model_name_or_path"], subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args["pretrained_model_name_or_path"], subfolder="scheduler")
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args["pretrained_model_name_or_path"], subfolder="text_encoder",
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args["pretrained_model_name_or_path"], subfolder="text_encoder_2",
    )
    vae_path = (
        args["pretrained_model_name_or_path"]
        if args["pretrained_vae_model_name_or_path"] is None
        else args["pretrained_vae_model_name_or_path"]
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        # subfolder="vae" if "pretrained_vae_model_name_or_path" in args else None,

    )
    unet = UNet2DConditionModel.from_pretrained(
        args["pretrained_model_name_or_path"], subfolder="unet",
    )

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if "use_ema" in args:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args["pretrained_model_name_or_path"], subfolder="unet",
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
    if "enable_npu_flash_attention" in args:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        #re enable
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=4,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)


    if "enable_xformers_memory_efficient_attention" in args:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            print("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if "use_ema" in args:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

    def load_model_hook(models, input_dir):
        if "use_ema" in args:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if "gradient_checkpointing" in args:
        unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if "allow_tf32" in args:
        torch.backends.cuda.matmul.allow_tf32 = True

    if "scale_lr" in args:
        args["learning_rate"] = (
                args["learning_rate"] * args["gradient_accumulation_steps"] * args[
            "train_batch_size"] * accelerator.num_processes
        )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if "use_8bit_adam" in args:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

        # Optimizer creation
    params_to_optimize = lora_layers.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args["learning_rate"],
        betas=(args["adam_beta1"], args["adam_beta2"]),
        weight_decay=args["adam_weight_decay"],
        eps=args["adam_epsilon"],
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Preprocessing the datasets.
    train_resize = transforms.Resize((args["resolutionH"] , args["resolutionW"]), interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop((args["resolutionH"] , args["resolutionW"]),) if "center_crop" in args else transforms.RandomCrop(
        (args["resolutionH"] , args["resolutionW"]))
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((1280, 960), interpolation=transforms.InterpolationMode.BILINEAR),
        # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ]
    )


    df1 = pd.read_csv('./newData/NewInput.csv')
    Samplesize = 30  # number of samples that you want
    df1.groupby(['category']).size()
    # remove categories where not enough samples
    df_remove_not_enought_examples = df1[~df1['category'].isin([1, 2, 22, 33, 34, 35])]
    df1 = df_remove_not_enought_examples.groupby('category', as_index=False).apply(
        lambda array: array.loc[np.random.choice(array.index, Samplesize, False), :])
    # df1 = df1.iloc[6478:6481, :]
    df1=  df1.reset_index(drop=True)
    index = df1.index.values
    df1['no'] = index.tolist()

    dataset = Dataset.from_pandas(df1)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    def find_between(s, first, last):
        try:
            start = s.index(first) + len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""

    def create_captions(path, list_no):
        subPaths = path.split("/")
        if (subPaths[3].find("-") > 0):
            text = find_between(subPaths[3], "@", "[")
            caption = text + " " + "poza" + " " + str(list_no)
        else:
            caption = subPaths[3] + " " + "poza" + " " + str(list_no)
        return caption

    def preprocess_train(examples):
        image_paths = [path.replace('\\', '/') for path in examples["path"]]
        image_numbers = [no for no in examples["no"]]
        images = [read_image(image_path, mode=ImageReadMode.RGB) for image_path in image_paths]
        captions = [create_captions(path, no) for path, no in zip(image_paths, image_numbers)]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.shape[1], image.shape[2]))
            image = train_resize(image)
            if "random_flip" in args  and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if "center_crop" in args:
                y1 = max(0, int(round((image.height - args["resolutionH"]) / 2.0)))
                x1 = max(0, int(round((image.width -  args["resolutionW"]) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args["resolutionH"], args["resolutionW"]))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        examples["captions"] = captions
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset.with_transform(preprocess_train)

        # Let's first compute all the embeddings so that we can free up the text encoders
        # from memory. We will pre-compute the VAE encodings too.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args["proportion_empty_prompts"],
        caption_column = "captions"
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        new_fingerprint_for_vae = Hasher.hash((vae_path, args))
        train_dataset_with_embeddings = train_dataset.map(
            compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
        )
        train_dataset_with_vae = train_dataset.map(
            compute_vae_encodings_fn,
            batched=True,
            batch_size=args["train_batch_size"],
            new_fingerprint=new_fingerprint_for_vae,
        )
        precomputed_dataset = concatenate_datasets(
            [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(['path', 'width', 'height', 'aspect_ratio', 'category', 'no'])], axis=1
        )
        precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)

    del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
    del text_encoders, tokenizers, vae
    gc.collect()
    if is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    def collate_fn(examples):
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

        # DataLoaders creation:

    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args["train_batch_size"],
        num_workers=args["dataloader_num_workers"],
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])
    if "max_train_steps" in args:
        args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=args["lr_warmup_steps"] * args["gradient_accumulation_steps"],
        num_training_steps=args["max_train_steps"] * args["gradient_accumulation_steps"],
        num_cycles=args["lr_num_cycles"],
        power=args["lr_power"],
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    if "use_ema" in args:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args["num_train_epochs"] = math.ceil(args["max_train_steps"] / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", args)

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args["pretrained_model_name_or_path"]:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Train!
    total_batch_size = args["train_batch_size"] * accelerator.num_processes * args["gradient_accumulation_steps"]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args["num_train_epochs"]}")
    logger.info(f"  Instantaneous batch size per device = {args["train_batch_size"]}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args["gradient_accumulation_steps"]}")
    logger.info(f"  Total optimization steps = {args["max_train_steps"]}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args["max_train_steps"]),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args["num_train_epochs"]):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Sample noise that we'll add to the latents
                model_input = batch["model_input"].to(accelerator.device)
                noise = torch.randn_like(model_input)
                if "noise_offset" in args:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args["noise_offset"] * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                bsz = model_input.shape[0]
                if not "timestep_bias_strategy" in args:
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                        model_input.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args["resolutionW"], args["resolutionH"])
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if "prediction_type" in args:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args["prediction_type"])

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if not "snr_gamma" in args:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args["snr_gamma"] * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args["train_batch_size"])).mean()
                train_loss += avg_loss.item() / args["gradient_accumulation_steps"]

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if  "use_ema" in args:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args["checkpointing_steps "] == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if "checkpoints_total_limit" in args  is not None:
                            checkpoints = os.listdir(args["output_dir"])
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args["checkpoints_total_limit"]:
                                num_to_remove = len(checkpoints) - args["checkpoints_total_limit"] + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args["output_dir"], f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args["max_train_steps"]:
                break

        if accelerator.is_main_process:
            if "validation_promp" in args   and epoch % args["validation_epochs"] == 0:
                logger.info(
                    f"Running validation... \n Generating {args["num_validation_images"]} images with prompt:"
                    f" {args.validation_prompt}."
                )
                if "use_ema" in args:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                # create pipeline
                vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder="vae" if "pretrained_vae_model_name_or_path" in args  else None,

                )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args["pretrained_vae_model_name_or_path"],
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    torch_dtype=weight_dtype,
                )
                if "prediction_type" in args:
                    scheduler_args = {"prediction_type": args["prediction_type"]}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                pipeline_args = {"prompt": args["validation_prompt"]}

                with autocast_ctx:
                    images = [
                        pipeline(**pipeline_args, generator=generator, num_inference_steps=25).images[0]
                        for _ in range(args["num_validation_images"])
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")

                del pipeline
                if is_torch_npu_available():
                    torch_npu.npu.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if "use_ema" in args:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args["output_dir"])
        if "use_ema" in args:
            ema_unet.copy_to(unet.parameters())

        # Serialize pipeline.
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if  not "pretrained_vae_model_name_or_path" in args   else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args["pretrained_vae_model_name_or_path"],
            unet=unet,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(args["output_dir"])
        if "prediction_type" in args:
            scheduler_args = {"prediction_type": args["prediction_type"]}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args["output_dir"])

        # run inference
        images = []
        if "validation_prompt" in args  and args["num_validation_images"] > 0:
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

            with autocast_ctx:
                images = [
                    pipeline(args["validation_prompt"], num_inference_steps=25, generator=generator).images[0]
                    for _ in range(args["num_validation_images"])
                ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")

    accelerator.end_training()

if __name__ == '__main__':
    args_as_dict = {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "pretrained_vae_model_name_or_path": "madebyollin/sdxl-vae-fp16-fix",
        "enable_xformers_memory_efficient_attention": True,
        # "center_crop": False,
        # "random_flip": False,
        "proportion_empty_prompts": 0.0,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "num_train_epochs": 2,
        "max_train_steps": 10000,
        "use_8bit_adam": True,
        "learning_rate": 5e-4,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "mixed_precision": "fp16",
        "validation_prompt": "a cute Sundar Pichai creature",
        "validation_epochs": 1,
        "checkpointing_steps": 5000,
        "output_dir": "sdxl-naruto-model",
        "logging_dir": "/log",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-08,
        "max_grad_norm": 1.0,
        "resolutionH": 1280,
        "resolutionW": 960,
        "dataloader_num_workers": 8,
        "lr_power":1,
        "lr_num_cycles":1
    }

    main(args_as_dict)
