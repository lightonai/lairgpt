import torch
import tokenizers
from typing import Union, Optional, List

from lairgpt.utils.remote import load_asset
from lairgpt.utils.assets import Config, Snapshot, Tokenizer
from lairgpt.gpt_model import LairGPT
from lairgpt.text_generator import TextGenerator


class PAGnol(TextGenerator):
    """Factory class for LairGPT preconfigured models."""

    def __init__(
        self,
        model: Union[torch.nn.Module, str],
        tokenizer: Union[tokenizers.Tokenizer, str],
        config=None,
        device: Optional[torch.device] = None,
        max_decoding_steps: int = 32,
    ):
        if isinstance(model, str):
            model = load_asset(model, "passed model snapshot")
        if isinstance(tokenizer, str):
            tokenizer = load_asset(tokenizer, "passed tokenizer")
        super().__init__(model, tokenizer, config)

    def __call__(
        self,
        inputs: Union[str, List[str]],
        mode: str = "nucleus",
        temperature: float = 1.0,
        k: int = 5,
        p: float = 0.9,
        max_decoding_steps: int = 32,
        skip_eos: bool = True,
    ) -> Optional[List[str]]:
        return super().infer(
            inputs,
            mode=mode,
            temperature=temperature,
            k=k,
            p=p,
            max_decoding_steps=max_decoding_steps,
            skip_eos=skip_eos,
        )

    @classmethod
    def small(cls):
        pagnol = cls(Snapshot.SMALL.value, Tokenizer.CCNET.value, Config.SMALL.value)
        return pagnol

    @classmethod
    def medium(cls):
        pagnol = cls(Snapshot.MEDIUM.value, Tokenizer.CCNET.value, Config.MEDIUM.value)
        return pagnol

    @classmethod
    def large(cls):
        pagnol = cls(Snapshot.LARGE.value, Tokenizer.CCNET.value, Config.LARGE.value)
        return pagnol
