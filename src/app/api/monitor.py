from pathlib import Path

from flask_restx import Namespace, Resource, fields

api = Namespace(Path(__file__).stem, "Monitoring related operations")


@api.route("/health")
class HealthCheck(Resource):
    @api.doc(description="Perform a health check")
    @api.marshal_with(
        api.model(
            'Health Check',
            {
                'success': fields.Boolean,
            },
        )
    )
    def get(self):
        return {'success': True}
