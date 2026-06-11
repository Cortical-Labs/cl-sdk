"""
Regression test for the "phantom" reward-burst defect.

Scenario (mirrors closed-loop pong):
  - A channel (the "paddle") is stimulated continuously via interrupt_then_stim,
    each tick issuing a sensory BURST that extends the channel's availability
    well ahead of the publish head.
  - Several multi-channel reward bursts are delivered with interrupt_then_stim.
    The reward's channel set SHARES the paddle channel.

Defect:
  A reward's multi-channel op stays in _stim_op_records for the ring-retention
  window (~5 s). A subsequent paddle interrupt_then_stim on the shared channel
  triggers a rebuild that re-simulates an already-published reward op. Because the
  shared channel's availability floor (the rebuild checkpoint) has advanced far
  ahead of the publish head, the reward's sync barrier drags the already-published
  burst FORWARD; the pulses that land at/after the publish head are written to the
  stim ring a SECOND time — a phantom duplicate burst.

This test spies on every StimRecord written to the shared ring and asserts that
each reward-only channel receives exactly one burst's worth of stims per reward,
never more.

Uses CL_SDK_ACCELERATED_TIME=1 so no real-time delay is needed.
"""
import os

import pytest

import cl


@pytest.fixture(autouse=True)
def _sim_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CL_SDK_VISUALISATION", "0")
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")

# Paddle channel — continuously stimulated, and ALSO part of the reward set.
_PADDLE_CHANNEL = 38
# Reward feedback channels (multi-channel → forces a rebuild). Includes the paddle
# channel to reproduce the shared-channel sync-barrier drag, plus reward-only
# channels we can check for duplication.
_REWARD_CHANNELS      = (5, 6, 13, 38, 45, 46)
_REWARD_ONLY_CHANNELS = tuple(c for c in _REWARD_CHANNELS if c != _PADDLE_CHANNEL)

_STIM_DURATION_US = 160
_STIM_CURRENT_UA  = 1

_BURST_COUNT = 10      # 100 Hz × 0.1 s
_BURST_HZ    = 100

# Sensory paddle burst: a full 1 s of availability ahead of the publish head so a
# re-simulated reward op gets dragged forward past the head (the phantom write).
_SENSORY_BURST_COUNT = 100
_SENSORY_BURST_HZ    = 100

_TICKS_PER_SECOND = 100
_STOP_AFTER_SECS  = 8
# Multiple rewards so an OLD reward op is still retained (< ring window) while a
# later multi-channel op re-populates the rebuild's affected-channel set and
# triggers a rebuild that re-simulates the old op.
_REWARD_AT_TICKS  = (50, 250, 450, 650)


def test_published_reward_burst_is_not_rewritten() -> None:
    written_per_channel: dict[int, int] = {}

    with cl.open() as neurons:
        paddle_design = cl.StimDesign(_STIM_DURATION_US, _STIM_CURRENT_UA)
        reward_design = cl.StimDesign(_STIM_DURATION_US, _STIM_CURRENT_UA)
        reward_burst  = cl.BurstDesign(_BURST_COUNT, _BURST_HZ)
        sensory_burst = cl.BurstDesign(_SENSORY_BURST_COUNT, _SENSORY_BURST_HZ)
        reward_set    = cl.ChannelSet(*_REWARD_CHANNELS)

        # Spy on every StimRecord written to the ring buffer.
        original_write_stims = neurons._shared_buffer.write_stims

        def spy_write_stims(records):
            for r in records:
                written_per_channel[r.channel] = written_per_channel.get(r.channel, 0) + 1
            return original_write_stims(records)

        neurons._shared_buffer.write_stims = spy_write_stims  # type: ignore[method-assign]

        for tick, _ in enumerate(
            neurons.loop(ticks_per_second=_TICKS_PER_SECOND, stop_after_seconds=_STOP_AFTER_SECS)
        ):
            # Continuous "paddle" sensory burst on the shared channel; extends the
            # channel's availability ~1 s ahead of the publish head.
            neurons.interrupt_then_stim(_PADDLE_CHANNEL, paddle_design, sensory_burst)
            # Periodic multi-channel reward bursts.
            if tick in _REWARD_AT_TICKS:
                neurons.interrupt_then_stim(reward_set, reward_design, reward_burst)

    # Each reward-only channel must receive exactly one burst of stim records per
    # reward — no more. A higher count means a published reward burst was re-written
    # to the ring (phantom duplicate).
    expected = len(_REWARD_AT_TICKS) * _BURST_COUNT
    for channel in _REWARD_ONLY_CHANNELS:
        written = written_per_channel.get(channel, 0)
        assert written == expected, (
            f"channel {channel}: expected exactly {expected} reward stim records "
            f"({len(_REWARD_AT_TICKS)} rewards × {_BURST_COUNT}) written to the ring, "
            f"but got {written}. More than {expected} indicates a published reward "
            f"burst was re-written (phantom duplicate)."
        )
