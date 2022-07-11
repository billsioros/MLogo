import base64
from pathlib import Path

import torch
from flask_restx import Namespace, Resource, fields
from transformers import DistilBertModel, DistilBertTokenizerFast

from mlogo.model import CNNGenerator

api = Namespace(Path(__file__).stem, "Model related operations")

parser = api.parser()
parser.add_argument('text', type=str, required=True, help='The description of the image')


@api.route("/generate")
class Predict(Resource):
    @api.doc(description="Generate a 256x256 image given a text")
    @api.expect(api.model('Prompt', {'text': fields.String}))
    @api.marshal_with(
        api.model(
            'Image',
            {
                'data': fields.String,
            },
        )
    )
    def post(self):
        text = api.payload['text']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = CNNGenerator.from_state(Path.cwd() / 'model' / 'generator.pt', device=device)

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
            image = image.cpu().detach().numpy().squeeze().reshape(256, 256, 3)

            return {'data': base64.b64encode(image)}
