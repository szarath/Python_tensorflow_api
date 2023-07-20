import os
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

api = Api(app)

# Load TensorFlow model
model = load_model('model.h5')

# Example data for users
users = [
    {"id": 1, "name": "John Doe", "age": 30},
    {"id": 2, "name": "Jane Smith", "age": 25},
    {"id": 3, "name": "Michael Johnson", "age": 40},
]

class HelloWorld(Resource):
    def get(self):
        return jsonify({'message': 'Hello, API!'})

class UserList(Resource):
    def get(self):
        return jsonify(users)

class UserDetails(Resource):
    def get(self, user_id):
        user = next((user for user in users if user["id"] == user_id), None)
        if user:
            return jsonify(user)
        else:
            return jsonify({"message": "User not found"}), 404

class Predict(Resource):
    def post(self):
        data = request.get_json()
        examples = data.get('examples', [])

        if not examples:
            return jsonify({"error": "No examples provided"}), 400

        # Perform prediction on each example
        predictions = []
        for example in examples:
            features = np.array(example).reshape(1, -1)
            prediction = model.predict(features)[0][0]
            predictions.append(prediction)

        return jsonify({"predictions": predictions})

# Add the API resources
api.add_resource(HelloWorld, '/hello')
api.add_resource(UserList, '/users')
api.add_resource(UserDetails, '/users/<int:user_id>')
api.add_resource(Predict, '/predict')

# Swagger UI configuration
SWAGGER_URL = '/swagger'
API_URL = '/swagger.json'

# OpenAPI Specification
openapi_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Example API",
        "version": "1.0.0"
    },
    "paths": {
        "/hello": {
            "get": {
                "summary": "Returns a greeting message",
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {
                                            "type": "string"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/users": {
            "get": {
                "summary": "Returns a list of users",
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {
                                                "type": "integer"
                                            },
                                            "name": {
                                                "type": "string"
                                            },
                                            "age": {
                                                "type": "integer"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/users/{user_id}": {
            "get": {
                "summary": "Returns details for a specific user",
                "parameters": [
                    {
                        "in": "path",
                        "name": "user_id",
                        "schema": {
                            "type": "integer"
                        },
                        "required": True,
                        "description": "ID of the user to retrieve"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "integer"
                                        },
                                        "name": {
                                            "type": "string"
                                        },
                                        "age": {
                                            "type": "integer"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "User not found"
                    }
                }
            }
        },
        "/predict": {
            "post": {
                "summary": "Makes predictions using provided examples",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "examples": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {
                                                "type": "number"
                                            }
                                        }
                                    }
                                },
                                "required": ["examples"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "predictions": {
                                            "type": "array",
                                            "items": {
                                                "type": "number"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid input"
                    }
                }
            }
        }
    }
}

# Save OpenAPI specification to swagger.json file
import json
with open("swagger.json", "w") as json_file:
    json.dump(openapi_spec, json_file)

# Define the function to serve the swagger.json
@app.route(API_URL)
def serve_swagger_json():
    return send_file("swagger.json")

# Swagger UI configuration
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Example API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# Print the current directory and files before starting the Flask app
print("Current Directory:", os.getcwd())
print("Files in the directory:", os.listdir())

if __name__ == '__main__':
    app.run(debug=True)
