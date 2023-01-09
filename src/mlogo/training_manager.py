"""The training manager"""


import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import trange
from transformers import DistilBertModel, DistilBertTokenizerFast

from mlogo.configuration import Configuration
from mlogo.model import CNNGenerator, MLogoDiscriminator


class TrainingManager:
    def __init__(self, configuration: Configuration) -> None:
        self._configuration = configuration
        self._device = configuration.device

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(self._device)

        def collate_batch(batch):
            captions, images = [], []
            for caption, image in batch:
                inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=56,
                    truncation=True,
                )

                input_ids = inputs["input_ids"].to(self._device)
                attention_mask = inputs["attention_mask"].to(self._device)

                with torch.no_grad():
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

                captions.append(outputs.last_hidden_state)
                images.append(image)

            captions = torch.stack(captions).to(self._device)
            images = torch.stack(images).to(self._device)

            return captions, images

        self._dataloader = torch.utils.data.DataLoader(
            configuration.dataset,
            batch_size=configuration.batch_size,
            shuffle=configuration.shuffle,
            num_workers=configuration.number_of_workers,
            collate_fn=collate_batch,
        )

    def __call__(self, load_state: bool = False, save_state: bool = False):
        self._configuration.cache_path.mkdir(parents=True, exist_ok=True)

        generator = (
            CNNGenerator.from_device(self._device)
            if load_state is False or self._configuration.cache_path is None
            else CNNGenerator.from_state(
                self._configuration.cache_path / 'generator.pt', device=self._device
            )
        )

        discriminator = (
            MLogoDiscriminator.from_device(self._device)
            if load_state is False or self._configuration.cache_path is None
            else MLogoDiscriminator.from_state(
                self._configuration.cache_path / 'discriminator.pt', device=self._device
            )
        )

        criterion = nn.BCELoss().to(self._device)
        generator_optimizer = SGD(generator.parameters(), lr=self._configuration.learning_rate)
        discriminator_optimizer = SGD(
            discriminator.parameters(), lr=self._configuration.learning_rate
        )

        def clause():
            generator.train()
            discriminator.train()

            generator_loss, discriminator_loss = 0, 0

            for captions, images in self._dataloader:
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ## Train with all-real batch
                discriminator.zero_grad()
                # Format batch
                label = torch.full((images.size(0),), 1, dtype=images.dtype, device=images.device)
                # Forward pass real batch through D
                output = discriminator(images).view(-1)
                # Calculate loss on all-real batch
                discriminator_error_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                discriminator_error_real.backward()

                ## Train with all-fake batch
                # Generate fake image batch with G
                fake = generator(captions)
                label.fill_(0)
                # Classify all fake batch with D
                output = discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                discriminator_error_fake = criterion(output, label)
                # Calculate the gradients for this batch
                discriminator_error_fake.backward()
                # Add the gradients from the all-real and all-fake batches
                # Update D
                discriminator_optimizer.step()

                discriminator_loss += (
                    discriminator_error_real + discriminator_error_fake
                ).item() / len(self._dataloader)

                # (2) Update G network: maximize log(D(G(z)))
                generator.zero_grad()
                label.fill_(1)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                discriminator_error_fake = criterion(output, label)
                # Calculate gradients for G, which propagate through the discriminator
                discriminator_error_fake.backward()
                # Update G
                generator_optimizer.step()

                generator_loss += discriminator_error_fake.item() / len(self._dataloader)

            return (generator_loss, discriminator_loss)

        with trange(self._configuration.iterations) as tqdm:
            avg_generator_loss, avg_discriminator_loss = 0, 0

            for _ in tqdm:
                epoch_generator_loss, epoch_discriminator_loss = clause()

                avg_generator_loss += epoch_generator_loss / self._configuration.iterations
                avg_discriminator_loss += epoch_discriminator_loss / self._configuration.iterations

                tqdm.set_postfix(
                    generator=avg_generator_loss,
                    discriminator=avg_discriminator_loss,
                )

        if save_state is True and self._configuration.cache_path is not None:
            generator.save(self._configuration.cache_path / 'generator.pt')
            discriminator.save(self._configuration.cache_path / 'discriminator.pt')
