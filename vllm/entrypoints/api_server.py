import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from typing import List
from transformers.generation.utils import GenerationConfig

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

# only support baichuan
@app.post("/chat")
async def chat(request: Request) -> Response: 
    tokenizer = engine.engine.tokenizer
    if tokenizer.__class__.__name__ != "BaichuanTokenizer": 
        ret = {"err": "chat not support in model {}".format(engine.engine.model_config.model)}
        return JSONResponse(ret)
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    messages = []
    messages.append({"role": "user", "content": prompt})
    input_ids = baichuan_chat(tokenizer, messages)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(None, sampling_params, request_id, input_ids)
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
    outputs = final_output.outputs
    print(outputs)
    # response = tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True)
    ret = {"text": outputs[0].text}
    return JSONResponse(ret)

def baichuan_chat(tokenizer, messages: List[dict]): 
    print(messages)
    generation_config = GenerationConfig.from_pretrained(
        engine.engine.model_config.model,
    )
    print(generation_config)
    max_new_tokens = generation_config.max_new_tokens
    print(tokenizer)
    max_input_tokens = tokenizer.model_max_length - max_new_tokens
    max_input_tokens = max(tokenizer.model_max_length // 2, max_input_tokens)
    total_input, round_input = [], []
    for i, message in enumerate(messages[::-1]):
        content_tokens = tokenizer.encode(message["content"])
        if message["role"] == "user":
            round_input = (
                [generation_config.user_token_id]
                + content_tokens
                + round_input
            )
            if (
                total_input
                and len(total_input) + len(round_input) > max_input_tokens
            ):
                break
            else:
                total_input = round_input + total_input
                if len(total_input) >= max_input_tokens:
                    break
                else:
                    round_input = []
        elif message["role"] == "assistant":
            round_input = (
                [generation_config.assistant_token_id]
                + content_tokens
                + [generation_config.eos_token_id]
                + round_input
            )
        else:
            raise ValueError(f"message role not supported yet: {message['role']}")
    total_input = total_input[-max_input_tokens:]  # truncate left
    total_input.append(generation_config.assistant_token_id)
    return total_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
