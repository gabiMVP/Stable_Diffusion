import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer
from lightning.pytorch.loggers import TensorBoardLogger as loggerPL
import bitsandbytes as bnb

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
save_dir = './weights/'
logs_dir = './lightning_logs/'


class DiffusionDataset(Dataset):
    def __init__(self, dataFrame, model_ID):
        # store the image and mask filepaths, and augmentation
        self.df = dataFrame
        self.weight_dtype = torch.float16
        self.imagePaths = dataFrame.loc[:, "path"]
        self.labels = dataFrame.loc[:, "category"]
        self.no = dataFrame.loc[:, "no"]
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            model_ID,
            subfolder="tokenizer",
            use_fast=False,

        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            model_ID,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        self.text_encoder1 = CLIPTextModel.from_pretrained(
            model_ID, subfolder="text_encoder"
        )
        self.text_encoder2 = CLIPTextModel.from_pretrained(
            model_ID, subfolder="text_encoder_2"
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((1280, 960), interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.imagePaths[idx].replace('\\', '/')
        image_no = self.no[idx]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self.transform(image)
        caption = self.create_captions(image_path, image_no)
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(caption)
        # input_ids = self.tokenize_captions(caption)
        # return {"pixel_values": pixel_values, "input_ids": input_ids}
        return image, prompt_embeds.squeeze(),pooled_prompt_embeds.squeeze()

    def find_between(self, s, first, last):
        try:
            start = s.index(first) + len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""

    def create_captions(self, path, list_no):
        subPaths = path.split("/")
        if (subPaths[3].find("-") > 0):
            text = self.find_between(subPaths[3], "@", "[")
            caption = text + " " + "poza" + " " + str(list_no)
        else:
            caption = subPaths[3] + " " + "poza" + " " + str(list_no)
        return caption

    # def tokenize_captions(self, caption):
    #
    #     inputs = self.tokenizer(
    #         caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
    #         return_tensors="pt"
    #     )
    #     return inputs.input_ids

    def encode_prompt(self,caption,  is_train=True):
        prompt_embeds_list = []
        text_encoders=[self.text_encoder1,self.text_encoder2]
        tokenizers =[self.tokenizer_one,self.tokenizer_two]
        with torch.no_grad():
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    caption,
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
                # pooled_prompt_embeds = prompt_embeds[0]
                pooled_prompt_embeds = prompt_embeds[1]
                prompt_embeds = prompt_embeds[-1][-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                # _, seq_len, bs_embed = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds.cpu() ,pooled_prompt_embeds.cpu()
        # return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}

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

class DiffusionDataModule(LightningDataModule):
    def __init__(self, dataframe, batch_size, model_id):
        super().__init__()
        self.train_df = dataframe
        self.batch_size = batch_size
        self.model_id = model_id

    def setup(self, stage=None):
        self.train_dataset = DiffusionDataset(
            self.train_df, self.model_id
        )
        self.test_dataset = DiffusionDataset(
            self.train_df, self.model_id
        )
        self.val_dataset = DiffusionDataset(
            self.train_df, self.model_id
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def val_dataloader(self):
        # no val dataset for diffusion
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        # no test dataset for diffusion
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size
        )


class LitningDiffusionModel(LightningModule):

    def __init__(self, model_ID,weight_dtype,VAE_ID):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            VAE_ID, subfolder="vae"

        )
        self.weight_dtype = weight_dtype
        self.unet = UNet2DConditionModel.from_pretrained(
            model_ID, subfolder="unet"
            # , addition_embed_type="text"
            # ,use_safetensors=True
            # ,low_cpu_mem_usage=False
            # ,device_map=None

        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_ID, subfolder="scheduler")

        # self.unet.enable_xformers_memory_efficient_attention()
        self.unet.requires_grad_(True)
        self.unet.train()
        self.vae.requires_grad_(False)

        self.unet.to(device, dtype=self.weight_dtype)
        self.vae.to(device, dtype=self.weight_dtype)


    def training_step(self, batch, batch_idx):
        image, prompt_embeds,pooled_prompt_embeds= batch
        batch_size =image.shape[0]
        inputs = image.to(device)
        prompt_embeds = prompt_embeds.to(device)
        latents = self.vae.encode(inputs).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents.to(device)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=self.weight_dtype)
        # Get the text embedding for conditioning
        # encoder_hidden_states = self.text_encoder(caption_input_ids, return_dict=False)[0]

        # XL things //that function does not do much torch.cat batch size
        add_time_ids = torch.cat(
            [self.compute_time_ids() for s in batch_size]
        ).squeeze()
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds.to(device, dtype=self.weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.squeeze()
        prompt_embeds = prompt_embeds.squeeze()
        pooled_prompt_embeds.to(device)
        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
        # model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]
        target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("train_loss", loss.item(), prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(params=self.unet.parameters(), lr=1e-6)
        optimizer = bnb.optim.AdamW8bit(params=self.unet.parameters(), lr=1e-6)

        return optimizer

    def forward(self, image_pixel, timeStep, caption_input_id):
        image_pixel.to(device)
        caption_input_id.to(device)
        timeStep.to(device)
        bs, ch, w, h = image_pixel.shape

        return self.unet(image_pixel, timeStep, caption_input_id, return_dict=False)[0]

    def compute_time_ids(self):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (1280, 960)
        original_size = (1280, 960)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids.unsqueeze(1)
        add_time_ids = add_time_ids.to(device, dtype=self.weight_dtype)
        return add_time_ids


def main():
    # model_ID = "CompVis/stable-diffusion-v1-4"
    # model_id = "stabilityai/stable-diffusion-2"
    from datasets import load_dataset

    model_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    VAE_NAME = "madebyollin/sdxl-vae-fp16-fix"
    df1 = pd.read_csv('./newData/NewInput.csv')
    # df1 = df1.iloc[6478:6678, :]
    index = df1.index.values
    df1['no'] = index.tolist()
    bacth_size = 1
    data_Module = DiffusionDataModule(df1, bacth_size, model_ID)
    data_Module.setup()
    weight_dtype=torch.float32
    VAE_NAME = model_ID
    model = LitningDiffusionModel(model_ID,weight_dtype,VAE_NAME)
    model.to(device)
    model.get_submodule('unet')


    image, prompt_embeds,pooled_prompt_embeds= next(iter(data_Module.train_dataloader()))
    batch_size =1
    inputs = image.to(device,dtype=model.weight_dtype)
    prompt_embeds = prompt_embeds.to(device)
    latents = model.vae.encode(inputs).latent_dist.sample()
    latents = latents * model.vae.config.scaling_factor
    latents.to(device)

        # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
        # Sample a random timestep for each image
    timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=model.weight_dtype)
    # Get the text embedding for conditioning
        # encoder_hidden_states = self.text_encoder(caption_input_ids, return_dict=False)[0]

        # XL things //that function does not do much torch.cat batch size
    add_time_ids = torch.cat(
        [model.compute_time_ids() for s in range(batch_size)]
    )
    # add_time_ids = torch.stack(
    #         [model.compute_time_ids() for s in range(batch_size)]
    #     )
        # unet_added_conditions = {"time_ids": add_time_ids}

    # prompt_embeds = prompt_embeds.squeeze()
    # pooled_prompt_embeds = pooled_prompt_embeds.squeeze()
    # add_time_ids = add_time_ids.squeeze()
    print(prompt_embeds.shape)
    print(pooled_prompt_embeds.shape)
    print(add_time_ids.shape)

    prompt_embeds.to(device, dtype=model.weight_dtype)
    pooled_prompt_embeds.to(device)

    unet_added_conditions = {"time_ids": add_time_ids}
    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
    # model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
    model_pred = model.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]
    target = noise
    # inputs = data_module.to(device)



    torch.set_float32_matmul_precision('medium')

    logger = loggerPL(save_dir=logs_dir, name="my_model")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="train_loss",
        mode="min",
        dirpath="content/weights",
        filename="sample-mnist-{epoch:02d}-{loss:.2f}",
    )

    trainer = Trainer(
        precision="16-mixed",
        max_epochs=2,
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=1,
        limit_val_batches=0,
        limit_test_batches=0,
    )
    trainer.fit(model, train_dataloaders=data_Module)
    new_model = LitningDiffusionModel.load_from_checkpoint(checkpoint_path="/content/stableDiff/stable.ckpt",
                                                           model_ID=model_ID)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_ID,
        text_encoder=new_model.text_encoder,
        vae=new_model.vae,
        unet=new_model.unet
    )


if __name__ == '__main__':
    main()
