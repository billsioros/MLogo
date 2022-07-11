from pathlib import Path

import click
import plotly.express as px
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

from mlogo.configuration import Configuration
from mlogo.dataset import MLogoDataset
from mlogo.model import CNNGenerator
from mlogo.training_manager import TrainingManager


@click.group()
def cli():
    """An ML approach to generating logos from text."""


@cli.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True),
    help="Where to load the dataset from.",
)
@click.option(
    "-s",
    "--state",
    type=click.Path(exists=False),
    help="Where to store the model's state to.",
)
def train(dataset, state):
    """Train the model."""

    dataset = MLogoDataset.from_directory(Path(dataset))

    configuration = Configuration(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        number_of_workers=0,  # TODO: The dataloader should be patched
        learning_rate=0.01,
        iterations=100,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        cache_path=Path(state),
    )

    manager = TrainingManager(configuration)

    manager(save_state=True)


@cli.command()
@click.option(
    "-l",
    "--load-state",
    type=click.Path(exists=True),
    required=True,
    help="Where to load the model's state from.",
)
@click.option(
    "-t",
    "--text",
    type=click.STRING,
    required=True,
    help="The input text to generate an image from.",
)
def generate(load_state, text):
    """Generate an image from a given text."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNNGenerator.from_state(load_state, device=device)

    model.eval()
    with torch.no_grad():
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            max_length=56,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)

        image = model(outputs.last_hidden_state)

        def as_grayscale_image(array, save_path=None):
            fig = px.imshow(array)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            if save_path is None:
                fig.show()
            else:
                with save_path.open("wb") as file:
                    fig.write_image(file)

        as_grayscale_image(
            image.cpu().detach().numpy().squeeze().reshape(256, 256, 3),
            save_path=Path.cwd() / f'{text}.png',
        )


if __name__ == '__main__':
    cli()
