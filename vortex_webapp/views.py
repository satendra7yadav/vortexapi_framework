from . import models
from django.shortcuts import render
from django.http import HttpResponse
import json
import ast
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

def home(request):
    return HttpResponse("Hello, Django!")
# Create your views here.

@api_view(['POST'])
def train(request):

    train_hyperparameters={}
    model = request.data.get('model')
    test_case = request.data.get('test_case')
    train_hyperparameters = ast.literal_eval(request.data.get('train_hyperparameters'))
    result = models.train_model(test_case,model,train_hyperparameters)
    json_output = json.dumps({"output" : result})
    return HttpResponse(json_output,content_type ="application/json")

@api_view(['POST'])
def predict(request):

    parameters = {}
    model = request.data.get('model')
    test_case = request.data.get('test_case') 
    parameters['time_step'] = request.data.get('time_step')
    parameters['time_step_precision'] = request.data.get('time_step_precision')
    parameters['total_time_steps'] = request.data.get('total_time_steps')
    result = models.predict_model(test_case,model,parameters)
    json_output = json.dumps({"output" : result})
    return HttpResponse(json_output,content_type ="application/json")