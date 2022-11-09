# Import modules
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_response

# Define flask app
app = Flask(__name__)
CORS(app)




# Define post functions
@app.post("/predict")
def predict():
    # Get user input in json format
    text = request.get_json().get('message')
    # TODO: check if text is valid

    response = get_response(text)
    message = {'Answer': response}

    # Response to json formar
    return jsonify(message)


# Run main app
if __name__ == '__main__':
    app.run(debug=True)