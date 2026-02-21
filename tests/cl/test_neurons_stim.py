import os
import pytest

from inline_snapshot import snapshot

import cl
from cl import Neurons, Stim, ChannelSet, StimDesign, BurstDesign, StimPlan

def test_channel_set():
    with pytest.raises(TypeError):
        channels = []
        channel_set = ChannelSet(*channels)

def test_stim_design():

    # Monophasic
    StimDesign(160, -1.0)

    # Biphasic
    StimDesign(160, -1.0, 160, 1.0)

    # Triphasic
    StimDesign(160, -1.0, 160, 1.0, 160, -1.0)

    with pytest.raises(ValueError):
        # Stim duration does not conform to duration bins
        StimDesign(150, -1.0)

    with pytest.raises(ValueError):
        # Stim current exceeds maximum recommended
        StimDesign(160, -StimDesign._CURRENT_LIMIT_UA - 0.1)

    with pytest.raises(ValueError):
        # Stim current has the same polarity across pulses
        StimDesign(160, -1.0, 160, -1.0)

def test_burst_design():

    burst_count       = 10
    burst_hz          = 100

    with pytest.raises(ValueError):
        # Negative burst_count
        BurstDesign(-burst_count, burst_hz)

    with pytest.raises(ValueError):
        # Negative burst_hz
        BurstDesign(burst_count, -burst_hz)

def test_stim():
    """
    We test the neurons.stim() function using three types of uses:
    - Stim call 1: legacy use without ChannelSet or StimDesign
    - Stim call 2: Stim without burst
    - Stim call 3: Stim with burst

    This includes a regression test that tests:
    - Stim queueing when stim is called on a busy channel
    - Future stims from stim bursts arriving at the correct tick iteration
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        frames_per_second   = neurons._frames_per_second

        lead_time_us        = 80
        lead_time_frames    = int(lead_time_us / 1e6 * frames_per_second)

        pulse_width_us      = 160
        biphasic_frames     = int(2 * pulse_width_us / 1e6 * frames_per_second)

        burst_hz            = 100
        inter_burst_frames  = int(1 / burst_hz * frames_per_second)

        start_timestamp     = neurons.timestamp()
        ticks_per_second    = 200
        tick_frames         = int(1 / ticks_per_second * frames_per_second)
        stop_after_ticks    = 4

        #
        # We deliver stims during the first tick then observe the stims
        # obtained through tick.analysis.stims. Compare this to the expected
        # stims for each tick
        #

        observed_tick_stims: dict[int, list[Stim]] = {}
        neurons_loop = neurons.loop(
            ticks_per_second=ticks_per_second,
            stop_after_ticks=stop_after_ticks
        )
        for tick in neurons_loop:

            now = neurons.timestamp()
            print(f"{now=} {tick.analysis.start_timestamp=} {now-tick.analysis.start_timestamp=}")

            if tick.iteration == 0:
                # (Stim call 1) Stim with legacy interface
                stim_channel    = 8
                stim_current_uA = 1.0
                neurons.stim(stim_channel, stim_current_uA, None, lead_time_us)

                # (Stim call 2) Stim without burst (single)
                channel_set = ChannelSet(8, 9)
                stim_design = StimDesign(pulse_width_us, -1.0, pulse_width_us, 1.0)
                neurons.stim(channel_set, stim_design, None, lead_time_us)

                # (Stim call 3) Stim with burst
                channel_set  = ChannelSet(16, 17)
                stim_design  = StimDesign(pulse_width_us, -1.0, pulse_width_us, 1.0)
                burst_design = BurstDesign(2, burst_hz)
                neurons.stim(channel_set, stim_design, burst_design, lead_time_us)

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel=} {stim.timestamp-tick.analysis.start_timestamp=}")

        # We should expect to see stims at every second tick iteration since
        # our tick rate is twice that of the burst rate
        expected_tick_stims: dict[int, list[Stim]] = {
            0: [],
            1: [
                Stim( # (Stim call 1)
                    channel=8,
                    timestamp=(
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                        )
                    ),
                Stim( # (Stim call 2)
                    channel=9,
                    timestamp=(
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                        )
                    ),
                Stim( # (Stim call 3)
                    channel=16,
                    timestamp=(
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                        )
                    ),
                Stim( # (Stim call 3)
                    channel=17,
                    timestamp=(
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                        )
                    ),
                Stim( # (Stim call 2) This is queued after Stim call 1
                    channel=8,
                    timestamp=(
                        start_timestamp
                        + tick_frames
                        + lead_time_frames # Stim call 1
                        + biphasic_frames  # Stim call 1
                        + lead_time_frames # Stim call 2
                        )
                    ),
            ],
            2: [],
            3: [
                Stim( # (Stim call 3)
                    channel=16,
                    timestamp=(
                        start_timestamp
                        + (3 * tick_frames)
                        + lead_time_frames
                        )
                    ),
                Stim( # (Stim call 3)
                    channel=17,
                    timestamp=(
                        start_timestamp
                        + (3 * tick_frames)
                        + lead_time_frames
                        )
                    ),
            ],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)

def test_stim_bursts():
    """
    We do tests for a range of burst frequencies to ensure stim intervals
    match that of reference that is collected from the device
    """
    import numpy as np

    burst_count          = 10
    ticks_per_second     = 10
    burst_frequencies_hz = [96, 100, 110, 151, 199]
    expected_frame_diffs = \
        {
            96:  [260, 261, 260, 261, 260, 261, 260, 261, 260, 11],
            100: [250, 250, 250, 250, 250, 250, 250, 250, 250, 10],
            110: [227, 228, 227, 228, 227, 228, 227, 228, 227, 11],
            151: [165, 166, 165, 166, 165, 166, 165, 166, 165, 11],
            199: [125, 126, 125, 126, 125, 126, 125, 126, 125, 11]
        }

    for burst_frequency in burst_frequencies_hz:
        stims = []
        with cl.open() as neurons:
            for tick in neurons.loop(ticks_per_second=ticks_per_second, stop_after_ticks=2):
                for stim in tick.analysis.stims:
                    stims.append(stim.timestamp)

                if tick.iteration == 0:
                    neurons.stim(8, -1, BurstDesign(burst_count, burst_frequency))
                    neurons.stim(8, -1)

        observed_frame_diffs = np.diff(stims)
        assert np.allclose(observed_frame_diffs, expected_frame_diffs[burst_frequency])

def test_invalid_stims():
    """
    We do tests for invalid stim parameters
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        with pytest.raises(ValueError):
            # Invalid lead time, < 80 us
            neurons.stim(1, 1, lead_time_us=79)

        with pytest.raises(ValueError):
            # Invalid lead time, not divisible by 80 us
            neurons.stim(1, 1, lead_time_us=90)

        with pytest.raises(ValueError):
            # Burst interval 40 us must be at least 80 us + duration 320 us
            neurons.stim(
                cl.ChannelSet(1),
                cl.StimDesign(160, -1.5, 160, 1.5),
                cl.BurstDesign(2, 25_000),
                lead_time_us=80
                )

def test_stim_plan():
    """
    We do the same test as test_stim() except with a StimPlan.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        frames_per_second = neurons._frames_per_second

        lead_time_us       = 80
        lead_time_frames   = int(lead_time_us / 1e6 * frames_per_second)

        pulse_width_us     = 160
        biphasic_frames    = int(2 * pulse_width_us / 1e6 * frames_per_second)

        burst_hz           = 100
        inter_burst_frames = int(1 / burst_hz * frames_per_second)

        start_timestamp    = neurons.timestamp()
        ticks_per_second   = 200
        tick_frames        = int(1 / ticks_per_second * frames_per_second)
        stop_after_ticks   = 4

        #
        # We build the stims initially as my_stim_plan, then run it
        # during the first tick.
        #

        my_stim_plan: StimPlan = neurons.create_stim_plan()
        # (Stim call 1) Stim with legacy interface
        stim_channel    = 8
        stim_current_uA = 1.0
        my_stim_plan.stim(stim_channel, stim_current_uA, None, lead_time_us)

        # (Stim call 2) Stim without burst (single)
        channel_set = ChannelSet(8, 9)
        stim_design = StimDesign(pulse_width_us, -1.0, pulse_width_us, 1.0)
        my_stim_plan.stim(channel_set, stim_design, None, lead_time_us)

        # (Stim call 3) Stim with burst
        channel_set  = ChannelSet(16, 17)
        stim_design  = StimDesign(pulse_width_us, -1.0, pulse_width_us, 1.0)
        burst_design = BurstDesign(2, burst_hz)
        my_stim_plan.stim(channel_set, stim_design, burst_design, lead_time_us)

        #
        # We deliver stims during the first tick then observe the stims
        # obtained through tick.analysis.stims. Compare this to the expected
        # stims for each tick
        #

        observed_tick_stims: dict[int, list[Stim]] = {}
        neurons_loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        for tick in neurons_loop:

            now = neurons.timestamp()
            print(f"{now=} {tick.analysis.start_timestamp=} {now-tick.analysis.start_timestamp=}")

            if tick.iteration == 0:
                my_stim_plan.run()

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel=} {stim.timestamp-tick.analysis.start_timestamp=}")

        # We should expect to see stims at every second tick iteration since
        # our tick rate is twice that of the burst rate
        expected_tick_stims: dict[int, list[Stim]] = {
            0: [],
            1: [
                Stim( # (Stim call 1)
                    channel   = 8,
                    timestamp = (
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                    )
                ),
                Stim( # (Stim call 2)
                    channel   = 9,
                    timestamp = (
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                    )
                ),
                Stim( # (Stim call 3)
                    channel   = 16,
                    timestamp = (
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                    )
                ),
                Stim( # (Stim call 3)
                    channel   = 17,
                    timestamp = (
                        start_timestamp
                        + tick_frames
                        + lead_time_frames
                    )
                ),
                Stim( # (Stim call 2) This is queued after Stim call 1
                    channel   = 8,
                    timestamp = (
                        start_timestamp
                        + tick_frames
                        + lead_time_frames # Stim call 1
                        + biphasic_frames  # Stim call 1
                        + lead_time_frames # Stim call 2
                    )
                ),
            ],
            2: [],
            3: [
                Stim( # (Stim call 3)
                    channel   = 16,
                    timestamp = (
                        start_timestamp
                        + (3 * tick_frames)
                        + lead_time_frames
                    )
                ),
                Stim( # (Stim call 3)
                    channel   = 17,
                    timestamp = (
                        start_timestamp
                        + (3 * tick_frames)
                        + lead_time_frames
                    )
                ),
            ],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)

def test_interrupt():
    """
    We test neurons.interrupt() by:
    - Running a neurons loop with tick frequency of 40 Hz
    - At each tick, we interrupt and send a new stim at frequencies of
      [40, 80, 120, 160] Hz. Each stim contains a burst of 1000 stims.
    - The expected output is:
        - Iteration 0: 0 stims detected
        - Iteration 1: 1 stims detected
        - Iteration 2: 2 stims detected
        - Iteration 3: 3 stims detected
        - Iteration 4: 4 stims detected
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        frames_per_second = neurons._frames_per_second

        lead_time_us      = 80
        lead_time_frames  = int(lead_time_us / 1e6 * frames_per_second)

        pulse_width_us    = 160
        biphasic_frames   = int(2 * pulse_width_us / 1e6 * frames_per_second)

        start_timestamp   = neurons.timestamp()
        ticks_per_second  = 40
        tick_frames       = int(1 / ticks_per_second * frames_per_second)
        stop_after_ticks  = 5

        #
        # We deliver stim bursts based on the burst_channels and burst_freqs
        # so both of these must have the same length. The position relates to
        # the associated tick.iteration.
        #
        # Avoid half frame frequencies in this test as we are using hand calculated
        # ground truth.
        #

        burst_channels     : list[int] = [8, 9, 10, 11]
        burst_freqs        : list[int] = [40, 50, 100, 200]
        inter_burst_frames : list[int] = [
            int(1 / freq * frames_per_second)
            for freq in burst_freqs
        ]

        observed_tick_stims: dict[int, list[Stim]] = {}
        neurons_loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        for tick in neurons_loop:

            now = neurons.timestamp()
            print(f"{now=} {tick.analysis.start_timestamp=} {now-tick.analysis.start_timestamp=}")

            neurons.interrupt(ChannelSet(*burst_channels))
            if tick.iteration < len(burst_channels):
                burst_channel = burst_channels[tick.iteration]
                burst_hz      = burst_freqs[tick.iteration]
                neurons.stim(
                    ChannelSet(burst_channel),
                    StimDesign(160, -1.0, 160, 1.0),
                    BurstDesign(1000, burst_hz),
                    )

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel} {stim.timestamp-tick.analysis.start_timestamp=}")

        # We should expect to see stims at every second tick iteration since
        # our tick rate is twice that of the burst rate
        expected_tick_stims: dict[int, list[Stim]] = {
            0: [],
            1: [
                Stim(
                    channel=burst_channels[0],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 1)
                        + lead_time_frames
                        )
                    ),
            ],
            2: [
                Stim(
                    channel=burst_channels[1],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 2)
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[1],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 2)
                        + lead_time_frames
                        + (inter_burst_frames[1] * 1)
                        )
                    ),
            ],
            3: [
                Stim(
                    channel=burst_channels[2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 3)
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 3)
                        + lead_time_frames
                        + (inter_burst_frames[2] * 1)
                        )
                    ),
                Stim(
                    channel=burst_channels[2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 3)
                        + lead_time_frames
                        + (inter_burst_frames[2] * 2)
                        )
                    ),
            ],
            4: [
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 1)
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 2)
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 3)
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 4)
                        )
                    ),
            ],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)

def test_interrupt_stimplan():
    """
    We do the same test as test_interrupt() except with a StimPlan.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        frames_per_second = neurons._frames_per_second

        lead_time_us      = 80
        lead_time_frames  = int(lead_time_us / 1e6 * frames_per_second)

        pulse_width_us    = 160
        biphasic_frames   = int(2 * pulse_width_us / 1e6 * frames_per_second)

        start_timestamp   = neurons.timestamp()
        ticks_per_second  = 40
        tick_frames       = int(1 / ticks_per_second * frames_per_second)
        stop_after_ticks  = 5


        #
        # We deliver stim bursts based on the burst_channels and burst_freqs
        # so both of these must have the same length. The position relates to
        # the associated tick.iteration.
        #
        # Avoid half frame frequencies in this test as we are using hand calculated
        # ground truth.
        #

        burst_channels     : list[int] = [8, 9, 10, 11]
        burst_freqs        : list[int] = [40, 50, 100, 200]
        inter_burst_frames : list[int] = [
            int(1 / freq * frames_per_second)
            for freq in burst_freqs
        ]

        #
        # We build the operations as a list of stim_plans for each tick iteration
        #
        stim_plans: list[StimPlan] = []
        for iteration in range(stop_after_ticks):
            stim_plan: StimPlan  = neurons.create_stim_plan()
            stim_plan.channels_to_interrupt = ChannelSet(*burst_channels)
            if iteration < len(burst_channels):
                burst_channel = burst_channels[iteration]
                burst_hz      = burst_freqs[iteration]
                stim_plan.stim(
                    ChannelSet(burst_channel),
                    StimDesign(160, -1.0, 160, 1.0),
                    BurstDesign(1000, burst_hz),
                    )
            stim_plans.append(stim_plan)

        observed_tick_stims: dict[int, list[Stim]] = {}
        neurons_loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        for tick in neurons_loop:

            now = neurons.timestamp()
            print(f"{now=} {tick.analysis.start_timestamp=} {now-tick.analysis.start_timestamp=}")

            stim_plans[tick.iteration].run()

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel} {stim.timestamp-tick.analysis.start_timestamp=}")

        # We should expect to see stims at every second tick iteration since
        # our tick rate is twice that of the burst rate
        expected_tick_stims: dict[int, list[Stim]] = {
            0: [],
            1: [
                Stim(
                    channel=burst_channels[0],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 1)
                        + lead_time_frames
                        )
                    ),
            ],
            2: [
                Stim(
                    channel=burst_channels[1],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 2)
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[1],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 2)
                        + lead_time_frames
                        + (inter_burst_frames[1] * 1)
                        )
                    ),
            ],
            3: [
                Stim(
                    channel=burst_channels[2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 3)
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 3)
                        + lead_time_frames
                        + (inter_burst_frames[2] * 1)
                        )
                    ),
                Stim(
                    channel=burst_channels[2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 3)
                        + lead_time_frames
                        + (inter_burst_frames[2] * 2)
                        )
                    ),
            ],
            4: [
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 1)
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 2)
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 3)
                        )
                    ),
                Stim(
                    channel=burst_channels[3],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 4)
                        + lead_time_frames
                        + (inter_burst_frames[3] * 4)
                        )
                    ),
            ],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)

def test_stimplan_run_at_timestamp_with_interrupt():
    """
    This test checks whether interrupt respects the burst frequency gap
    across several combinations of frequencies, using:
    - StimPlan.run(at_timestamp)
    - StimPlan.channels_to_interrupt

    Expected output:
    ```
    40 120 [625 625]  [625 625 625 208 209 208] [625 625 625 208 209 208]
    40 100 [625 625]  [625 625 625 250 250 250] [625 625 625 250 250 250]
    40 60 [625 625]   [625 625 625 416 417 416] [625 625 625 416 417 416]
    60 120 [416 417]  [416 417 416 208 209 208] [416 417 416 208 209 208]
    60 100 [416 417]  [416 417 416 250 250 250] [416 417 416 250 250 250]
    60 40 [416 417]   [416 417 416 625 625 625] [416 417 416 625 625 625]
    100 120 [250 250] [250 250 250 208 209 208] [250 250 250 208 209 208]
    100 60 [250 250]  [250 250 250 416 417 416] [250 250 250 416 417 416]
    100 40 [250 250]  [250 250 250 625 625 625] [250 250 250 625 625 625]
    120 100 [208 209] [209 208 209 250 250 250] [209 208 209 250 250 250]
    120 60 [208 209]  [209 208 209 416 417 416] [209 208 209 416 417 416]
    120 40 [208 209]  [209 208 209 625 625 625] [209 208 209 625 625 625]
    ```

    Note: on the real system, half-frame interrupt may cause an extra 1 frame gap. Such as:
    ```
    60 120 [416 417] [416 417 417 208 209 208]
                          ^^^ ^^^
    ```
    """
    import numpy as np

    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"

    channel_set = ChannelSet(8)

    with cl.open() as neurons:
        for rate_1 in [40, 60, 100, 120]:
            for rate_2 in [120, 100, 60, 40]:
                if rate_1 == rate_2:
                    continue

                plan_1 = neurons.create_stim_plan()
                plan_1.channels_to_interrupt = channel_set
                plan_1.stim(channel_set, -1, burst_design=BurstDesign(1000, rate_1))

                plan_2 = neurons.create_stim_plan()
                plan_2.channels_to_interrupt = channel_set
                plan_2.stim(channel_set, -1, burst_design=BurstDesign(4, rate_2))

                stims = []
                for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=1):

                    for stim in tick.analysis.stims:
                        stims.append(stim.timestamp)

                    if tick.iteration == 0:
                        now = neurons.timestamp()
                        plan_1.run(at_timestamp=now + 5000)
                        plan_2.run(at_timestamp=now + 15000)

                neurons.interrupt(channel_set)

                # Stims: ... A [Gap A] A [Gap A] A [ Gap A] B [Gap B] B [Gap B] B [Gap B] B
                stim_ts_diff = np.diff(stims)

                rate_1_interval_us = int((1_000_000 / rate_1 / 20) + 0.5) * 20
                rate_2_interval_us = int((1_000_000 / rate_2 / 20) + 0.5) * 20

                rate_1_total_duration_us = rate_1_interval_us * 1000
                rate_2_total_duration_us = rate_2_interval_us * 4

                rate_1_times_us  = np.arange(0, rate_1_total_duration_us, step=rate_1_interval_us)
                rate_1_timestamps, rate_1_interval_remainders = np.divmod(rate_1_times_us, 40)

                rate_2_times_us  = np.arange(0, rate_2_total_duration_us, step=rate_2_interval_us)
                rate_2_timestamps, rate_2_interval_remainders = np.divmod(rate_2_times_us, 40)

                # Interrupt rate 1 and switch to rate 2
                interrupt_point    = (rate_1_timestamps < 10000).sum()
                rate_2_timestamps += rate_1_timestamps[interrupt_point]
                expected_ts        = np.concat([rate_1_timestamps[:interrupt_point], rate_2_timestamps])
                expected_ts_diff   = np.diff(expected_ts)

                print(rate_1, rate_2, stim_ts_diff[:2], stim_ts_diff[-6:], expected_ts_diff[-6:])
                assert np.all(np.equal(expected_ts_diff[-6], stim_ts_diff[-6]))

def test_neurons_sync():
    """
    This tests Neurons.sync() by performing two groups of stims on two channels
    with burst stim of different frequencies.

    Without sync, group 2 stims will occur immediately following group 1.
    With sync, group 2 will wait until the end of the slowest frequency in group 1.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:

        stim_design    = StimDesign(160, -1.0, 160, 1.0)

        channel_set_1  = ChannelSet(8)
        burst_design_1 = BurstDesign(2, 100)    # Interval of 250 frames

        channel_set_2  = ChannelSet(10)
        burst_design_2 = BurstDesign(2, 20)     # Interval of 1250 frames

        group_1_stims = []
        for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=1):

            if tick.iteration == 0:
                # Group 1
                neurons.stim(channel_set_1, stim_design, burst_design_1)
                neurons.stim(channel_set_2, stim_design, burst_design_2)

                # Group 2
                neurons.sync(channel_set_1 | channel_set_2)
                neurons.stim(channel_set_1, stim_design)

            for stim in tick.analysis.stims:
                if stim.channel == 8:
                    group_1_stims.append(stim.timestamp)

        group_gap = group_1_stims[-1] - group_1_stims[0]
        print(group_gap, group_1_stims)
        assert len(group_1_stims) == 3
        assert group_gap >= 1250

def test_stimplan_sync():
    """
    This tests sync functionality by:
    - Running plan_1 with stim bursts at three different frequencies.
    - Interrupt plan_1 with plan_2 at a point that does not align well with the frequencies.
    - plan_2 sync then one stim on the requested channels.
    - All plan_2 stims should occur at the same, which is at the time
      of the slowest channel to become available.

    Expected output for `print(plan_2_sync_frame, last_stim_timestamp, last_stim_channels)`

    ```
    12502 12502 [8, 9, 10, 16, 17]
    ```
    """
    from collections import defaultdict
    from pprint import pprint

    import numpy as np

    channel_frequencies_hz = \
        {
            8: 20,      # Interval: 1250 frames
            9: 40,      # Interval: 625  frames
            10: 100     # Interval: 250  frames
        }

    plan_1_channels    = list(channel_frequencies_hz.keys())
    plan_2_channels    = plan_1_channels + [16, 17]

    plan_1_channel_set = ChannelSet(*plan_1_channels)
    plan_2_channel_set = ChannelSet(*plan_2_channels)

    sample_frequency        = 25_000
    plan_1_start_frames     = 5000                                              # Relative to "now", to be calculated later
    plan_1_duration_frames  = 7345                                              # Set a duration that does not divide evenly with frequencies
    plan_2_start_frames     = plan_1_start_frames + plan_1_duration_frames
    plan_2_sync_frame       = plan_2_start_frames                               # When do we expect sync to occur, this will be set later

    with cl.open() as neurons:

        plan_1 = neurons.create_stim_plan()
        plan_1.channels_to_interrupt = plan_1_channel_set
        for channel, frequency_hz in channel_frequencies_hz.items():
            burst_design = BurstDesign(1000, frequency_hz)
            plan_1.stim(ChannelSet(channel), -1, burst_design=burst_design)

            # Calculate the number of frames this frequency would persist
            # if interrupted at end of plan_1_duration_frames
            interval_frames      = sample_frequency // frequency_hz
            num_stims, remainder = divmod(plan_1_duration_frames, interval_frames)
            expected_duration    = (num_stims * interval_frames) + ((remainder > 0) * interval_frames)
            plan_2_sync_frame    = max(plan_2_sync_frame, plan_1_start_frames + expected_duration)

        plan_2 = neurons.create_stim_plan()
        plan_2.channels_to_interrupt = plan_2_channel_set
        plan_2.sync(plan_2_channel_set)
        plan_2.stim(plan_2_channel_set, -1)

        stims = defaultdict(list)
        now   = neurons.timestamp()
        for tick in neurons.loop(ticks_per_second=10, stop_after_ticks=11):

            if tick.iteration == 0:
                now = neurons.timestamp()
                plan_1.run(at_timestamp=now + plan_1_start_frames)
                plan_2.run(at_timestamp=now + plan_2_start_frames)

            for stim in tick.analysis.stims:
                stims[stim.timestamp - now].append(stim.channel)

        # Fetch the stims of the largest timestamp,
        # which should be all of plan_2 if sync is working correctly
        last_stim_timestamp = max(stims.keys())
        last_stim_channels  = stims[last_stim_timestamp]
        plan_2_sync_frame  += 2                                                   # Add 2 frames for minimum lead time

        pprint(stims)
        print(plan_2_sync_frame, last_stim_timestamp, last_stim_channels)
        assert np.allclose(plan_2_sync_frame, last_stim_timestamp, atol=1)        # In case of half-frame alignment
        assert np.allclose(np.sort(plan_2_channels), np.sort(last_stim_channels))