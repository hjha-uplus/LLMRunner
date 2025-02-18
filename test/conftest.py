from PIL import Image
from pytest import fixture

from src.llm_runner.util import decode_image


# 8 by 8 grid image
DEMO_BASE64_STR = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="

TEMP_TEXT = "Describe this image"


@fixture
def pil_image(
    demo_base64_str=DEMO_BASE64_STR,
):
    return decode_image(demo_base64_str)


@fixture
def messages(
    pil_image: Image,
    text: str = TEMP_TEXT,
):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]