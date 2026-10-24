"""Tests that code_rl threads grade_truncated_response from the CLI down to each env."""

import asyncio
from unittest.mock import patch

import pytest

from tinker_cookbook.recipes.code_rl.code_env import (
    DeepcoderDatasetBuilder,
    DeepcoderEnvGroupBuilder,
)
from tinker_cookbook.recipes.code_rl.deepcoder_tool import DeepcoderTask
from tinker_cookbook.recipes.code_rl.train import CLIConfig
from tinker_cookbook.rl.message_env import EnvFromMessageEnv

_TASK = DeepcoderTask(
    problem="Print 42.",
    tests=[{"input": "", "output": "42", "testtype": "stdin_stdout", "metadata": {}}],
)


def _build_datasets(grade_truncated_response: bool | None = None) -> list[DeepcoderEnvGroupBuilder]:
    """Run DeepcoderDatasetBuilder on one task and return every env group builder.

    With grade_truncated_response=None the builder's default is used.
    """
    if grade_truncated_response is None:
        builder = DeepcoderDatasetBuilder(
            model_name_for_tokenizer="Qwen/Qwen3-8B",
            renderer_name="qwen3",
            batch_size=1,
            group_size=2,
        )
    else:
        builder = DeepcoderDatasetBuilder(
            model_name_for_tokenizer="Qwen/Qwen3-8B",
            renderer_name="qwen3",
            batch_size=1,
            group_size=2,
            grade_truncated_response=grade_truncated_response,
        )
    with patch(
        "tinker_cookbook.recipes.code_rl.code_env.load_deepcoder_tasks", return_value=[_TASK]
    ):
        train, test = asyncio.run(builder())
    assert test is not None
    builders = [
        b
        for b in [*train.get_batch(0), *test.get_batch(0)]
        if isinstance(b, DeepcoderEnvGroupBuilder)
    ]
    # One train builder and one test builder.
    assert len(builders) == 2
    return builders


@pytest.mark.parametrize("flag", [True, False])
def test_flag_reaches_train_and_test_builders(flag: bool):
    builders = _build_datasets(grade_truncated_response=flag)

    assert all(b.grade_truncated_response is flag for b in builders)


def test_flag_is_off_by_default():
    assert all(b.grade_truncated_response is False for b in _build_datasets())
    assert CLIConfig().grade_truncated_response is False


@pytest.mark.parametrize("flag", [True, False])
def test_flag_reaches_envs(flag: bool):
    group_builder = _build_datasets(grade_truncated_response=flag)[0]

    envs = asyncio.run(group_builder.make_envs())

    assert len(envs) == 2
    assert all(isinstance(env, EnvFromMessageEnv) for env in envs)
    assert all(
        isinstance(env, EnvFromMessageEnv) and env.grade_truncated_response is flag for env in envs
    )
