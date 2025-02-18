import time
import requests
import threading
from typing import List, Any, Dict

from src.llm_runner.api import run_server
from src.llm_runner.util import messages_to_serialiable_data


def test_server(
    messages: List[Dict[str, Any]],
    HOST_NAME: str = "0.0.0.0",
    PORT: int = 8000,
):
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(0.5)

    _message = messages_to_serialiable_data(messages)
    response = requests.get(f"http://{HOST_NAME}:{PORT}/")
    assert response.status_code == 200
    print("GET / =>", response.status_code, response.json())

    response = requests.post(f"http://{HOST_NAME}:{PORT}/invoke", data=str({"message": _message[0]}))
    assert response.status_code == 200
    print("POST /invoke =>", response.status_code, response.json())

    response = requests.post(f"http://{HOST_NAME}:{PORT}/batch", data=str(_message))
    assert response.status_code == 200
    print("POST /batch =>", response.status_code, response.json())
