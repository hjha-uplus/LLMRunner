import io
import base64
from typing import Union, List, Dict, Any

from PIL import Image


def encode_image(image: Union[str, Image.Image]) -> str:
    if isinstance(image, Image.Image):
        byte_image = io.BytesIO()
        image.save(byte_image, format="PNG")
        return base64.b64encode(byte_image.getvalue()).decode('utf-8')
    else:
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def decode_image(encoded_string: str) -> Image:
    image_data = base64.b64decode(encoded_string)
    return Image.open(io.BytesIO(image_data))


def messages_to_serialiable_data(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = [message for message in messages] * 5
    return [
        {
            "role": message["role"],
            "content": [
                {
                    "type": "image",
                    "image": encode_image(message["content"][0]["image"]),
                },
                {
                    "type": "text",
                    "text": message["content"][1]["text"],
                },
            ],
        }
        for message in messages
    ]

def serialized_data_to_messages(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [
                {
                    "type": "image",
                    "image": decode_image(message["content"][0]["image"]),
                },
                {
                    "type": "text",
                    "text": message["content"][1]["text"],
                },
            ],
        }
        for message in data
    ]



if __name__ == "__main__":
    # 8 by 8 grid image
    encoded_string = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="
    # 10 by 10 red
    # encoded_string = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC="
    decoded_image = decode_image(encoded_string)
    print("decoded_image.size", decoded_image.size)
    decoded_image.save("decoded_image.png")
