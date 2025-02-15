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

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
save_dir = './weights/'
logs_dir = './lightning_logs/'


class DiffusionDataset(Dataset):
    def __init__(self, dataFrame, model_ID):
        # store the image and mask filepaths, and augmentation
        self.df = dataFrame
        self.imagePaths = dataFrame.loc[:, "path"]
        self.labels = dataFrame.loc[:, "category"]
        self.no = dataFrame.loc[:, "no"]
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_ID, subfolder="tokenizer",
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
        input_ids = self.tokenize_captions(caption)
        # return {"pixel_values": pixel_values, "input_ids": input_ids}
        return image, input_ids

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

    def tokenize_captions(self, caption):

        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids


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
    def __init__(self, model_ID):
        super().__init__()
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_ID, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(
            model_ID, subfolder="vae"
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            model_ID, subfolder="unet"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_ID, subfolder="text_encoder"
        )
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()

        self.unet.requires_grad_(True)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)

    def training_step(self, batch, batch_idx):
        inputs, caption_input_ids = batch
        inputs = inputs.to(device)
        caption_input_ids = caption_input_ids.to(device)
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
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(caption_input_ids, return_dict=False)[0]
        # pdb.set_trace()
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("train_loss", loss.item(), prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.unet.parameters(), lr=1e-6)
        return optimizer

    def forward(self, image_pixel, timeStep, caption_input_id):
        image_pixel.to(device)
        caption_input_id.to(device)
        timeStep.to(device)
        bs, ch, w, h = image_pixel.shape
        latents = self.vae.encode(image_pixel).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents.to(device)
        return self.unet(latents, timeStep, caption_input_id, return_dict=False)[0]

def main():
    model_ID = "CompVis/stable-diffusion-v1-4"
    model_id = "stabilityai/stable-diffusion-2"
    train = True
    if train:
        df1 = pd.read_csv('./newData/NewInput.csv')
        # df1 = df1.iloc[6478:6678, :]
        index = df1.index.values
        df1['no'] = index.tolist()
        bacth_size = 2
        data_Module = DiffusionDataModule(df1, bacth_size, model_ID)
        data_Module.setup()
        # pixel_values, tokenized_caption = next(iter(data_Module.train_dataloader()))
        # print(pixel_values.shape)
        # print(tokenized_caption.shape)
        torch.set_float32_matmul_precision('medium')

        logger = loggerPL(save_dir=logs_dir, name="my_model")
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="train_loss",
            mode="min",
            dirpath="content/weights",
            filename="stable_dif-{epoch:02d}-{train_loss:.2f}",
        )

        trainer = Trainer(
            precision="16-mixed",
            max_epochs=5,
            accumulate_grad_batches=4,
            callbacks=[checkpoint_callback],
            logger=logger,
            accelerator="cpu",
            devices=1,
            limit_val_batches=0,
            limit_test_batches=0,
        )
        model = LitningDiffusionModel(model_ID)

        # model.train()
        model.vae.requires_grad = False
        model.unet.requires_grad = True
        model.text_encoder.requires_grad = False
        model.text_encoder.eval()
        model.unet.train()
        model.to(device)
        # model()

        trainer.fit(model, train_dataloaders=data_Module)
        trainer.save_checkpoint("stable.ckpt")

    df1 = pd.read_csv('./newData/NewInput.csv')
    # df1 = df1.iloc[6478:6678, :]
    index = df1.index.values
    df1['no'] = index.tolist()
    bacth_size = 2
    data_Module = DiffusionDataModule(df1, bacth_size, model_ID)
    data_Module.setup()

    s = 1
    model = LitningDiffusionModel(model_ID)
    model.to(device)
    model.get_submodule('unet')
    useNormalNotXL=False
    if useNormalNotXL:
        pixel_values, tokenized_caption = next(iter(data_Module.train_dataloader()))
        latents = model.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * model.vae.config.scaling_factor
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = model.text_encoder(tokenized_caption, output_hidden_states=True, return_dict=False)[0]
        model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False )[0]
        print(model_pred)

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
