from flask import Flask, jsonify
from flask import request, Response
import sys
import json
from types import SimpleNamespace
import os
from http import HTTPStatus
from datetime import datetime
from datetime import timedelta

script_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(script_path, "..")
sys.path.append(code_path)

app = Flask(__name__)

@app.route("/")
def instructions():
    return ('<p>instructions:<p>')    

@app.route("/lpr-export", methods = ["POST"])
def run_export(): # Renamed function for clarity
    return jsonify("This is a test endpoint"), HTTPStatus.GONE
 
@app.route('/status', methods=['GET'])
def health_check():
	return "OK"
