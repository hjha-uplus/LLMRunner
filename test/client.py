import requests
from typing import List, Any, Dict

from src.llm_runner.util import messages_to_serialiable_data


def test_client(
    messages: List[Dict[str, Any]],
    HOST_NAME: str = "0.0.0.0",
    PORT: int = 8000,
):
    
    _message = messages_to_serialiable_data(messages)
    try:
        response = requests.get(f"http://{HOST_NAME}:{PORT}/")
        assert response.status_code == 200
        print("GET / =>", response.status_code, response.json())

        up = True
    except requests.exceptions.RequestException:
        up = False
    
    if up:
        response = requests.post(f"http://{HOST_NAME}:{PORT}/invoke", data=str({"message": _message[0]}))
        assert response.status_code == 200
        print("POST /invoke =>", response.status_code, response.json())

        response = requests.post(f"http://{HOST_NAME}:{PORT}/batch", data=str(_message))
        assert response.status_code == 200
        print("POST /batch =>", response.status_code, response.json())

