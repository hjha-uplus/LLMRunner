import ast
import uvicorn
import threading
import time
import requests
from typing import List, Dict, Any

from PIL import Image
from fastapi import FastAPI, Request

from .model import BaseLLMModel
from .util import serialized_data_to_messages


def create_app(llm_model: BaseLLMModel = BaseLLMModel()):
    app = FastAPI()

    @app.get("/")
    async def read_root():
        return {"Hello": "World"}

    @app.post("/invoke")
    async def invoke(request: Request):
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
        data = ast.literal_eval(body_str)
        input_message = data.get("message")
        input_messages = serialized_data_to_messages([input_message])
        result = llm_model.invoke(input_messages[0])
        return {"result": result}

    @app.post("/batch")
    async def batch(request: Request):
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
        input_messages = ast.literal_eval(body_str)
        input_messages = serialized_data_to_messages(input_messages)
        results = llm_model.batch(input_messages)
        return {"result": results}

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs,
):
    app = create_app(**kwargs)
    uvicorn.run(app, host=host, port=port)
