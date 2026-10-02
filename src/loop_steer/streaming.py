"""Stream Transformers generation while keeping the model available for hooks."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from threading import Event

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


def stream_chat(model, tokenizer, messages, *, max_new_tokens=256, temperature=1.0, top_p=0.7):
    """Yield generated text; propagate worker errors and stop on interruption.

    Hooks installed on ``model`` run in the generation thread. Keep their
    handles alive until this iterator finishes, then remove them.
    """
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1.0,
    )
    stop = Event()

    class StopOnInterrupt(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return stop.is_set()

    def generate():
        # Inference mode is thread-local, so enter it inside the worker.
        with torch.inference_mode():
            return model.generate(
                **inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature, "top_p": top_p} if temperature > 0 else {}),
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnInterrupt()]),
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate)
        try:
            while True:
                try:
                    yield next(streamer)
                except Empty:
                    if future.done():
                        future.result()  # Raise the original generation error.
                        raise RuntimeError("Generation ended without closing the text stream")
                except StopIteration:
                    break
            future.result()
        finally:
            stop.set()
