from typing import Union, Optional
import gc
import torch
from base import LanguageModel
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


class VLLM(LanguageModel):
    def __init__(
        self,
        model_pth: str,
        num_gpus: int,
        max_length: int,
        dtype: str = "bfloat16",
        gpu_mem: float = 0.95,
        swap_space: float = 16,
        max_new_tokens=None,
        **kwargs,
    ):
        super().__init__()
        """
        Args:
            model_pth (str): The path to the directory containing the pre-trained model.
            max_new_tokens (int, optional): The maximum number of new tokens to generate during inference. Defaults to None.
            max_length (int, optional): The maximum length of the input sequence. Defaults to 2048.
        """
        assert dtype in ["bfloat16", "float32"]

        self.model = LLM(
            model=model_pth,
            dtype=dtype,
            trust_remote_code=True,  # may be useful
            max_model_len=max_length,  # total sequence length
            gpu_memory_utilization=gpu_mem,  # default 0.9
            tensor_parallel_size=num_gpus,
            swap_space=swap_space,
        )
        self.max_new_tokens = max_new_tokens

        if model_pth.count("/") == 1:
            # from hub
            name = model_pth.split("/")[-1].lower()
            is_chat = "inst" in name
            # we do not support inst LLMs for simplicity
            assert is_chat is False

    def generate(
        self,
        inputs: Union[str, list[str], list[dict], list[list[dict]]],
        n: int = 1,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        stop: Union[None, str, list[str]] = None,
        **kwargs,
    ):

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            max_tokens=(
                self.max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
        )

        outputs = self.model.generate(inputs, sampling_params)

        return outputs

    def close(self):
        # Delete the llm object and free the memory:
        # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2076870351
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
