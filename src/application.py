from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import the CORS module
from src.pipeline.predict_pipeline import generate_translation
import os
import sys
from src.exception import CustomException
from src.logger import logging


application = Flask(__name__)
app = application
CORS(app)  # Enable CORS for all routes

# a
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    user_input = request.args.get('input', '')
    data = generate_translation(50, source_sentence=(user_input,))
    print(data[0][:10])
    print(type(str(data)))


    return jsonify(str(data[0][:10]))

if __name__ == '__main__':
    app.run()
