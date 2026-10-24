"""Tests for Renderer.parse_truncated_response using real tokenizers.

Response tokens are built the way a sampler would produce them: render the
conversation with build_supervised_example, drop the generation prompt prefix, then
remove the stop token and cut the remaining tokens at chosen points.
"""

from __future__ import annotations

import pytest

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    get_text_content,
)
from tinker_cookbook.renderers.testing_utils import skip_if_deepseek_tokenizer_bug
from tinker_cookbook.tokenizer_utils import get_tokenizer

REASONING = "Let me draft print(41) and check it."
ANSWER = "The final answer is print(42) and nothing else."

# (model_name, renderer_name, has_reasoning, has_tool_calls)
RENDERERS = [
    ("Qwen/Qwen3-8B", "qwen3", True, True),
    ("Qwen/Qwen3-8B", "qwen3_disable_thinking", False, True),
    ("Qwen/Qwen3-8B", "qwen3_instruct", False, True),
    ("Qwen/Qwen3.5-35B-A3B", "qwen3_5", True, True),
    ("Qwen/Qwen3.5-35B-A3B", "qwen3_5_disable_thinking", False, True),
    ("meta-llama/Llama-3.2-1B-Instruct", "llama3", False, False),
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3", False, True),
    ("deepseek-ai/DeepSeek-V3.1", "deepseekv3_thinking", True, False),
    ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning", True, True),
    ("moonshotai/Kimi-K2-Thinking", "kimi_k2", True, True),
    ("moonshotai/Kimi-K2.5", "kimi_k25", True, True),
    ("moonshotai/Kimi-K2.5", "kimi_k25_disable_thinking", False, True),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3", True, True),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3_disable_thinking", False, True),
]
RENDERER_IDS = [renderer_name for _, renderer_name, _, _ in RENDERERS]

_USER: Message = {"role": "user", "content": "Write the program."}


def _make_renderer(model_name: str, renderer_name: str) -> Renderer:
    skip_if_deepseek_tokenizer_bug(model_name)
    return get_renderer(renderer_name, get_tokenizer(model_name))


def _sampled_tokens(renderer: Renderer, assistant: Message) -> list[int]:
    """Tokens a sampler returns for *assistant*, without the trailing stop token."""
    prompt = renderer.build_generation_prompt([_USER]).to_ints()
    full = renderer.build_supervised_example([_USER, assistant])[0].to_ints()
    assert full[: len(prompt)] == prompt, "generation prompt is not a prefix of the example"
    response = full[len(prompt) :]
    stop_tokens = renderer.get_stop_sequences()
    while response and response[-1] in stop_tokens:
        response = response[:-1]
    return response


def _cut_after(renderer: Renderer, tokens: list[int], marker: str) -> list[int]:
    """Shortest prefix of *tokens* whose decoded text contains *marker*."""
    for end in range(1, len(tokens) + 1):
        if marker in str(renderer.tokenizer.decode(tokens[:end])):
            return tokens[:end]
    raise AssertionError(f"{marker!r} not found in {renderer.tokenizer.decode(tokens)!r}")


def _answer_message(has_reasoning: bool) -> Message:
    if has_reasoning:
        return {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": REASONING},
                {"type": "text", "text": ANSWER},
            ],
        }
    return {"role": "assistant", "content": ANSWER}


@pytest.mark.parametrize("model_name,renderer_name,has_reasoning,_", RENDERERS, ids=RENDERER_IDS)
def test_cut_inside_answer_returns_answer_only(
    model_name: str, renderer_name: str, has_reasoning: bool, _: bool
):
    renderer = _make_renderer(model_name, renderer_name)
    tokens = _sampled_tokens(renderer, _answer_message(has_reasoning))

    message = renderer.parse_truncated_response(_cut_after(renderer, tokens, "print(42)"))

    assert message is not None
    answer = get_text_content(message)
    assert "print(42)" in answer
    assert "and nothing else" not in answer
    assert "print(41)" not in answer, "reasoning leaked into the answer text"
    assert "<think>" not in answer and "</think>" not in answer
    assert not message.get("tool_calls")


@pytest.mark.parametrize("model_name,renderer_name,has_reasoning,_", RENDERERS, ids=RENDERER_IDS)
def test_whole_response_without_stop_token(
    model_name: str, renderer_name: str, has_reasoning: bool, _: bool
):
    """A response cut exactly where the stop token would go parses like a stopped one."""
    renderer = _make_renderer(model_name, renderer_name)
    tokens = _sampled_tokens(renderer, _answer_message(has_reasoning))

    message = renderer.parse_truncated_response(tokens)

    assert message is not None
    assert get_text_content(message).strip() == ANSWER


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [(m, r) for m, r, has_reasoning, _ in RENDERERS if has_reasoning],
    ids=[r for _, r, has_reasoning, _ in RENDERERS if has_reasoning],
)
def test_cut_inside_reasoning_returns_none(model_name: str, renderer_name: str):
    renderer = _make_renderer(model_name, renderer_name)
    tokens = _sampled_tokens(renderer, _answer_message(has_reasoning=True))

    assert renderer.parse_truncated_response(_cut_after(renderer, tokens, "print(41)")) is None


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [(m, r) for m, r, has_reasoning, _ in RENDERERS if has_reasoning],
    ids=[r for _, r, has_reasoning, _ in RENDERERS if has_reasoning],
)
def test_cut_before_any_answer_text_returns_none(model_name: str, renderer_name: str):
    renderer = _make_renderer(model_name, renderer_name)
    tokens = _sampled_tokens(renderer, _answer_message(has_reasoning=True))
    # Cut just after the reasoning ends: the answer has not started yet.
    reasoning_end = "<|end|>" if renderer_name.startswith("gpt_oss") else "</think>"

    assert renderer.parse_truncated_response(_cut_after(renderer, tokens, reasoning_end)) is None


@pytest.mark.parametrize(
    "model_name,renderer_name,has_reasoning",
    [(m, r, has_reasoning) for m, r, has_reasoning, has_tools in RENDERERS if has_tools],
    ids=[r for _, r, _, has_tools in RENDERERS if has_tools],
)
def test_tool_call_returns_none(model_name: str, renderer_name: str, has_reasoning: bool):
    """A truncated turn that contains a tool call, complete or not, is not an answer."""
    renderer = _make_renderer(model_name, renderer_name)
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(name="check_solution", arguments='{"code": "print(1)"}'),
        id="call_0",
    )
    assistant: Message = {
        "role": "assistant",
        "content": (
            [
                {"type": "thinking", "thinking": REASONING},
                {"type": "text", "text": "Checking."},
            ]
            if has_reasoning
            else "Checking."
        ),
        "tool_calls": [tool_call],
    }
    tokens = _sampled_tokens(renderer, assistant)

    assert renderer.parse_truncated_response(tokens) is None
    assert renderer.parse_truncated_response(_cut_after(renderer, tokens, "print(1)")) is None


def test_role_colon_cut_inside_answer():
    renderer = _make_renderer("meta-llama/Llama-3.2-1B-Instruct", "role_colon")
    tokens = renderer.tokenizer.encode(" The final answer is print(42)", add_special_tokens=False)

    message = renderer.parse_truncated_response(tokens)

    assert message is not None
    assert get_text_content(message) == "The final answer is print(42)"


def test_empty_response_returns_none():
    renderer = _make_renderer("Qwen/Qwen3-8B", "qwen3")

    assert renderer.parse_truncated_response([]) is None


class _MinimalRenderer(Renderer):
    """A custom renderer that does not override parse_truncated_response."""

    def get_stop_sequences(self) -> list[int]:
        return [0]

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        raise NotImplementedError

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return {"role": "assistant", "content": "anything"}, True


def test_renderers_that_do_not_opt_in_return_none():
    renderer = _MinimalRenderer(get_tokenizer("Qwen/Qwen3-8B"))

    assert renderer.parse_truncated_response([1, 2, 3]) is None
