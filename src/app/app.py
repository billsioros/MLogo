from flask import Flask
from flask_restx import Api

from app.api.model import api as model
from app.api.monitor import api as monitor


def create_app():
    app = Flask(__name__)

    api = Api(
        app,
        title='MLogo',
        version='1.0',
        description='An ML approach to generating logos from text',
        prefix="/api/v1",
    )

    api.add_namespace(monitor)
    api.add_namespace(model)

    return app


if __name__ == '__main__':
    create_app()
