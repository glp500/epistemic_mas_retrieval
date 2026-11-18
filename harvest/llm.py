"""Load one pinned model and generate text: batched, with or without thinking."""


def on_cpu(inner):
    """Wrap a module so it runs on CPU; hidden from .to() so it is never moved."""
    import torch

    class OnCPU(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return self.__dict__["inner"](x.cpu(), *args, **kwargs).to(x.device)

    wrapper = OnCPU()
    wrapper.__dict__["inner"] = inner
    return wrapper


def presence_penalty(penalty, start):
    """vLLM's presence penalty, which Transformers lacks: every token already in the
    reply (columns from `start` on) loses `penalty` from its logit, once."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class Presence(LogitsProcessor):
        def __call__(self, input_ids, scores):
            seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, input_ids[:, start:], True)
            return scores - penalty * seen

    return LogitsProcessorList([Presence()])


class Model:
    """`config` is the protocol merged with one model's entry. batch_size 1 (the
    v3 default) gives the original one-prompt-at-a-time behaviour, unpadded."""

    def __init__(self, config, device=None):
        from transformers import AutoModelForImageTextToText

        pinned = self.setup(config, device)
        torch = self.torch
        dtype = torch.float32 if self.device == "cpu" else getattr(torch, config["dtype"])
        spread = config.get("device_map")  # "auto" shares one model across the job's GPUs
        placement = {"device_map": spread} if spread else {}
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["model_id"], dtype=dtype, attn_implementation=config["attention"], **pinned, **placement
        )
        # Large lookups (Gemma per-layer embeddings) and unused towers can stay on CPU.
        for path in config.get("cpu_modules", []):
            parent, name = path.rsplit(".", 1)
            setattr(self.model.get_submodule(parent), name, on_cpu(self.model.get_submodule(path)))
        if not spread:
            self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.metadata = {
            "backend": "transformers",
            "model_id": config["model_id"],
            "revision": config["revision"],
            "device": self.device,
            "dtype": "float32" if self.device == "cpu" else config["dtype"],
            "attention": config["attention"],
            "batch_size": self.batch_size,
            "decoding": self.decoding,
            "thinking_decoding": self.thinking_decoding,
            "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
        }

    def setup(self, config, device):
        """Settings, chat template and tokenizer, shared by both backends. Returns the pin."""
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        torch.set_num_threads(2)
        self.device = device or config["device"]
        self.max_tokens = config["max_prompt_tokens"]
        self.batch_size = config.get("batch_size", 1)
        # Per model: `decoding` with thinking off, `thinking_decoding` with it on (defaults to the same).
        self.decoding = config.get("decoding", {"do_sample": False})
        self.thinking_decoding = config.get("thinking_decoding", self.decoding)
        self.marks = config.get("thinking")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use a GPU node, or --device cpu for a slow check.")
        pinned = {"revision": config["revision"], "local_files_only": config["local_files_only"]}
        self.processor = AutoProcessor.from_pretrained(config["model_id"], **pinned)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        return pinned

    def reseed(self, value):
        """Sampling draws from torch's global generator. Seeding it per episode or screen
        cell makes a rerun on the same hardware and batch shapes give the same draws."""
        self.torch.manual_seed(value)

    def peak_gb(self):
        """Peak GPU memory so far, over all this process's GPUs (includes the weights)."""
        torch = self.torch
        return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9

    def prefix(self, prompt, thinking=False):
        """Chat-template string. `prompt` is one user message or a list of messages.
        enable_thinking is always explicit: the Qwen3.5 defaults differ by size."""
        messages = prompt
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )

    def inputs(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, add_special_tokens=False)
        if inputs["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Prompt exceeds configured token limit; never truncate task evidence silently")
        return inputs.to(self.device)

    def complete(self, prefixes, max_new_tokens, decoding=None):
        """Continue each prefix. Returns [{ids, finished, prompt_tokens}], ids cut at
        the first end token; `finished` is False if the token limit stopped it."""
        settings = dict(self.decoding if decoding is None else decoding)
        penalty = settings.pop("presence_penalty", 0)
        results = []
        for start in range(0, len(prefixes), self.batch_size):
            inputs = self.inputs(prefixes[start : start + self.batch_size])
            width = inputs["input_ids"].shape[1]
            extra = {"logits_processor": presence_penalty(penalty, width)} if penalty else {}
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **settings, **extra)
            lengths = inputs["attention_mask"].sum(1).tolist()
            for row, length in zip(output[:, width:].tolist(), lengths):
                end = next((i for i, token in enumerate(row) if token in self.eos), None)
                results.append(
                    {"ids": row if end is None else row[:end], "finished": end is not None, "prompt_tokens": length}
                )
        return results

    def generate(self, prompts, max_new_tokens):
        """Thinking off. Returns [{answer, prompt_tokens}], one per prompt."""
        done = self.complete([self.prefix(p) for p in prompts], max_new_tokens)
        return [
            {"answer": self.tokenizer.decode(r["ids"], skip_special_tokens=True), "prompt_tokens": r["prompt_tokens"]}
            for r in done
        ]

    def think(self, prompts, budget, answer_tokens):
        """Thinking on. A reply that uses up `budget` gets its thought closed (if
        still open) and is continued for `answer_tokens` more, so every player
        answers. Returns [{thought, answer, thought_tokens, truncated, prompt_tokens}]."""
        from harvest.prompts import split_thought

        prefixes = [self.prefix(p, thinking=True) for p in prompts]
        replies = self.complete(prefixes, budget, self.thinking_decoding)
        texts = [self.tokenizer.decode(r["ids"], skip_special_tokens=False) for r in replies]
        waiting = [i for i, r in enumerate(replies) if not r["finished"]]
        extras = dict.fromkeys(waiting, "")
        if waiting:
            resumed = []
            for i in waiting:
                closed = self.marks["close"] in texts[i]
                resumed.append(prefixes[i] + texts[i] + ("" if closed else self.marks["resume"]))
            for i, r in zip(waiting, self.complete(resumed, answer_tokens, self.thinking_decoding)):
                extras[i] = self.tokenizer.decode(r["ids"], skip_special_tokens=False)
        results = []
        for i, text in enumerate(texts):
            prefilled = prefixes[i].rstrip().endswith(self.marks["open"])
            thought, answer, closed = split_thought(text, self.marks, prefilled)
            answer += extras.get(i, "")
            results.append(
                {
                    "thought": thought,
                    "answer": self.clean(answer),
                    "thought_tokens": len(self.tokenizer(thought, add_special_tokens=False)["input_ids"]),
                    "truncated": not closed,
                    "prompt_tokens": replies[i]["prompt_tokens"],
                }
            )
        return results

    def clean(self, text):
        """Drop leftover special tokens from an answer decoded with them kept."""
        for token in self.tokenizer.all_special_tokens:
            text = text.replace(token, "")
        return text.strip()

    def score_options(self, prompt, options):
        """Log-probability of replying with exactly each option (thinking off): the
        sum over the option's tokens plus its end token, in fp32."""
        torch = self.torch
        prefix = self.prefix(prompt)
        end = self.marks["end"]
        base = len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])
        scores = []
        for start in range(0, len(options), self.batch_size):
            chunk = options[start : start + self.batch_size]
            texts = [prefix + option + end for option in chunk]
            sizes = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) - base for t in texts]
            inputs = self.inputs(texts)
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False, logits_to_keep=max(sizes) + 1).logits.float()
            ids = inputs["input_ids"]
            for row, size in enumerate(sizes):
                log_p = logits[row, -size - 1 : -1].log_softmax(-1)
                scores.append(float(log_p.gather(1, ids[row, -size:, None]).sum()))
        return scores


class VLLMModel(Model):
    """The same interface on vLLM, used when the model's config has a `vllm:` section
    (its keys go to vllm.LLM). Every prompt in a call is scheduled together, so
    `batch_size` is unused. Each request gets its own sampling seed, drawn in order
    from the seed set by `reseed`, so draws don't depend on which prompts share a batch."""

    def __init__(self, config, device=None):
        import random

        import vllm
        from transformers import AutoConfig, GenerationConfig

        pinned = self.setup(config, device)
        if self.device != "cuda":
            raise RuntimeError("The vLLM backend needs a GPU")
        self.vllm = vllm
        self.seeds = random.Random(0)
        self.llm = vllm.LLM(
            model=config["model_id"], revision=config["revision"], tokenizer_revision=config["revision"],
            dtype=config["dtype"], seed=0, **config["vllm"],
        )
        try:  # the end tokens Transformers would use: generation_config.json, else config.json
            generation = GenerationConfig.from_pretrained(config["model_id"], **pinned)
        except OSError:
            generation = GenerationConfig.from_model_config(AutoConfig.from_pretrained(config["model_id"], **pinned))
        eos = generation.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.metadata = {
            "backend": "vllm",
            "vllm_version": vllm.__version__,
            "vllm": config["vllm"],
            "model_id": config["model_id"],
            "revision": config["revision"],
            "device": self.device,
            "dtype": config["dtype"],
            "decoding": self.decoding,
            "thinking_decoding": self.thinking_decoding,
            "gpu": self.torch.cuda.get_device_name(),
        }

    def reseed(self, value):
        import random

        self.seeds = random.Random(value)

    def peak_gb(self):
        """GPU memory in use on this device now. vLLM reserves most of it at start-up
        (`gpu_memory_utilization`), so this is the reservation, not a measured peak."""
        free, total = self.torch.cuda.mem_get_info()
        return (total - free) / 1e9

    def token_prompts(self, texts):
        prompts = []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > self.max_tokens:
                raise ValueError("Prompt exceeds configured token limit; never truncate task evidence silently")
            prompts.append({"prompt_token_ids": ids})
        return prompts

    def complete(self, prefixes, max_new_tokens, decoding=None):
        settings = dict(self.decoding if decoding is None else decoding)
        sampled = settings.pop("do_sample", False)
        shared = settings if sampled else {"temperature": 0.0}  # the remaining keys are SamplingParams fields
        params = [
            self.vllm.SamplingParams(
                max_tokens=max_new_tokens, stop_token_ids=sorted(self.eos), skip_special_tokens=False,
                seed=self.seeds.randrange(2**31), **shared,
            )
            for _ in prefixes
        ]
        prompts = self.token_prompts(prefixes)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        results = []
        for prompt, output in zip(prompts, outputs):
            row = list(output.outputs[0].token_ids)
            end = next((i for i, token in enumerate(row) if token in self.eos), None)
            finished = end is not None or output.outputs[0].finish_reason == "stop"
            results.append({"ids": row if end is None else row[:end], "finished": finished, "prompt_tokens": len(prompt["prompt_token_ids"])})
        return results

    def score_options(self, prompt, options):
        """As Model.score_options, from vLLM's log-probabilities of the prompt's own tokens.
        One option per call: these logits span the whole prompt and vocabulary (about
        1 GB per prompt), memory that vLLM's start-up reservation doesn't set aside."""
        prefix = self.prefix(prompt)
        end = self.marks["end"]
        base = len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])
        params = self.vllm.SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        scores = []
        for option in options:
            prompts = self.token_prompts([prefix + option + end])
            output = self.llm.generate(prompts, params, use_tqdm=False)[0]
            ids = prompts[0]["prompt_token_ids"]
            scores.append(sum(output.prompt_logprobs[i][ids[i]].logprob for i in range(base, len(ids))))
        return scores


def load_model(config, device=None):
    """vLLM when the model's config has a `vllm:` section, Transformers otherwise."""
    return VLLMModel(config, device) if config.get("vllm") else Model(config, device)
