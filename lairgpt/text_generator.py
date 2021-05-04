import torch
import torch.nn.functional as F
import tokenizers
from typing import List, Union, Dict, Optional
from .utils.mask import get_autoregressive_mask
from .gpt_model import LairGPT


class TextGenerator:
    def __init__(
        self,
        model: Union[torch.nn.Module, str],
        tokenizer: Union[tokenizers.Tokenizer, str],
        config=None,
        device: Optional[torch.device] = None,
        max_decoding_steps: int = 32,
    ):
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")

        if isinstance(model, torch.nn.Module):
            self.model = model
        else:
            assert config is not None, "trying to load a model from path without providing a config"
            self.model = LairGPT(**config)
            self.model.load_state_dict(torch.load(model))

        self.model = self.model.to(self.device)

        if isinstance(tokenizer, str):
            self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", direction="right"
        )
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_ids = self.tokenizer.token_to_id("<EOS>")
        self.pad_ids = self.tokenizer.token_to_id("<PAD>")

        self.max_seq_len = config["max_seq_len"]
        self.max_decoding_steps = max_decoding_steps

        self.decode = {
            "greedy": self.greedy_decoding,
            "topk": self.topk_decoding,
            "nucleus": self.nucleus_decoding,
        }

        self.temperature = 1.0
        self.k = 5
        self.p = 0.9

    def greedy_decoding(self, logits: torch.Tensor) -> torch.LongTensor:
        logits = logits / self.temperature
        next_tokens = torch.argmax(logits, dim=1)
        return next_tokens

    def topk_decoding(self, logits: torch.Tensor) -> torch.LongTensor:
        logits = logits / self.temperature
        values, idx = torch.topk(logits, self.k, dim=1)
        logits[logits < values[:, -1].view(-1, 1)] = -float("inf")
        probs = F.softmax(logits, dim=1)
        next_token = torch.multinomial(probs, num_samples=1).view(-1)
        return next_token

    def nucleus_decoding(self, logits: torch.Tensor) -> torch.LongTensor:
        logits = logits / self.temperature
        probs = F.softmax(logits, dim=1)
        values, idx = probs.sort(descending=True, dim=1)
        for i, (v, index) in enumerate(zip(values, idx)):
            remove_idx = index[v.cumsum(dim=-1) > self.p]
            if remove_idx.shape[0] == self.vocab_size:
                mask = logits[i, :] < v[0]
                logits[i, mask] = -float("inf")
            else:
                logits[i, remove_idx] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).view(-1)
        return next_token

    def _get_targets(self, input_ids: torch.LongTensor):
        return input_ids.size(1) - torch.sum(((input_ids == self.pad_ids) * 1), dim=1) - 1

    def _get_ended_examples(self, pred_ids: torch.LongTensor) -> torch.BoolTensor:
        return pred_ids == self.eos_ids

    def _update_mapping(
        self, current_ids_to_init_ids: Dict[int, int], ended_examples: torch.BoolTensor
    ):
        n_ended = 0
        for current_ids, has_ended in enumerate(ended_examples):
            if has_ended.item() is True:
                current_ids_to_init_ids.pop(current_ids - n_ended, None)
                current_ids_to_init_ids = {
                    (k - 1 if k > current_ids - n_ended else k): v
                    for k, v in current_ids_to_init_ids.items()
                }
                n_ended += 1
        assert list(current_ids_to_init_ids.keys()) == list(range(len(current_ids_to_init_ids)))

        return current_ids_to_init_ids

    @torch.no_grad()
    def infer(
        self,
        inputs: Union[str, List[str]],
        mode: str = "nucleus",
        temperature: float = 1.0,
        k: int = 5,
        p: float = 0.9,
        max_decoding_steps: int = 32,
        skip_eos: bool = True,
    ) -> Optional[List[str]]:

        self.max_decoding_steps = max_decoding_steps
        self.temperature = temperature
        self.k = k
        self.p = p

        if isinstance(inputs, list):
            inputs = [i.strip() for i in inputs]
        else:
            inputs = inputs.strip()

        self.model.eval()
        assert mode in self.decode
        if isinstance(inputs, str):
            inputs = [inputs]
        input_ids = torch.tensor(
            [
                encoding.ids[: self.max_seq_len - max_decoding_steps - 1]
                for encoding in self.tokenizer.encode_batch(inputs)
            ]
        )
        batch_size = input_ids.size(0)

        if input_ids.size(1) > self.max_seq_len:
            print("Context size too long.. Example dropped")
            return None

        targets = self._get_targets(input_ids)
        input_ids, targets = (x.to(self.device) for x in [input_ids, targets])

        current_ids_to_init_ids: Dict[int, int] = {x: x for x in range(batch_size)}
        decoded_ids: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(self.max_decoding_steps):
            logits = self.model(
                input_ids, attn_mask=get_autoregressive_mask(input_ids.size(1)).to(self.device)
            )
            targeted_logits = torch.vstack([t[idx, :] for t, idx in zip(logits, targets)])
            if skip_eos:
                targeted_logits[:, self.eos_ids] = -float("inf")

            pred_ids = self.decode[mode](targeted_logits)

            # Update decoded ids
            for i, ids in enumerate(pred_ids.tolist()):
                decoded_ids[current_ids_to_init_ids[i]] += [ids]

            # Update state
            ended_examples = self._get_ended_examples(pred_ids)
            input_ids = input_ids[~ended_examples, :]
            pred_ids = pred_ids[~ended_examples]
            targets = targets[~ended_examples] + 1
            batch_size = input_ids.size(0)

            if (
                batch_size == 0 or input_ids.size(1) + 1 > self.max_seq_len
            ):  # all found <eos> or max_seq_len reach
                break

            current_ids_to_init_ids = self._update_mapping(current_ids_to_init_ids, ended_examples)

            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([self.pad_ids] * batch_size).to(self.device).view(-1, 1),
                ],
                dim=1,
            )
            input_ids = input_ids.scatter_(1, targets.view(-1, 1), pred_ids.view(-1, 1))

        return [self.tokenizer.decode(ids).strip() for ids in decoded_ids]
