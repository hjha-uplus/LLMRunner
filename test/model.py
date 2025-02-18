from typing import Any, Dict, List

from src.llm_runner.model import get_model, get_inputs_from_processor


def test_model(
    messages: List[Dict[str, Any]],
):
    model, processor, process_vision_info = get_model()

    inputs = get_inputs_from_processor(
        messages,
        processor,
        process_vision_info,
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
