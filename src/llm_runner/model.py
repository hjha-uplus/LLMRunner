from abc import ABC
from typing import Tuple, Callable, List, Dict, Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLProcessor
)


def get_model(
    qwen_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
) -> Tuple[Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Callable]:
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    return model, processor, process_vision_info


def get_inputs_from_processor(
    messages: List[Dict[str, Any]],
    processor: Qwen2VLProcessor,
    process_vision_info: Callable,
) -> torch.Tensor:
    # Preparation for inference
    input_texts = [
        processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs: torch.Tensor = processor(
        text=input_texts,
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


class BaseLLMModel(ABC):
    def invoke(self, input_message: Dict[str, Any]) -> List[str]:
        pass

    def batch(self, input_messages: List[Dict[str, Any]]) -> List[str]:
        pass


class LLMModel(BaseLLMModel):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model, self.processor, self.process_vision_info = get_model(model_name)

    def _run_with_input(self, inputs: Dict[str, Any]) -> List[str]:
        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4196,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text

    def invoke(self, input_message: Dict[str, Any]) -> List[str]:
        inputs = get_inputs_from_processor(
            [input_message],
            self.processor,
            self.process_vision_info,
        )
        inputs = inputs.to("cuda")

        return self._run_with_input(inputs)

    def batch(self, input_messages: List[Dict[str, Any]]) -> List[str]:
        inputs = get_inputs_from_processor(
            input_messages,
            self.processor,
            self.process_vision_info,
        )
        inputs = inputs.to("cuda")

        return self._run_with_input(inputs)
