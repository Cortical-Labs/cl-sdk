import os
import pytest

from inline_snapshot import snapshot

import cl
from cl import Neurons, Stim, ChannelSet, StimDesign, BurstDesign
from cl.stim_plan import StimPlan

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
        StimDesign(160, -StimDesign._MAX_CURRENT - 0.1)

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
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
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
            print(f"{now=} {tick.timestamp=} {now-tick.timestamp=}")

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
                print(f"\t{stim.timestamp=} {stim.channel=} {stim.timestamp-tick.timestamp=}")

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

def test_invalid_stims():
    """
    We do tests for invalid stim parameters
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
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
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
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
            print(f"{now=} {tick.timestamp=} {now-tick.timestamp=}")

            if tick.iteration == 0:
                my_stim_plan.run()

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel=} {stim.timestamp-tick.timestamp=}")

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
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
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

        burst_channels     : list[int] = [8, 9, 10, 11]
        burst_freqs        : list[int] = [40, 80, 120, 160]
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
            print(f"{now=} {tick.timestamp=} {now-tick.timestamp=}")

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
                print(f"\t{stim.timestamp=} {stim.channel} {stim.timestamp-tick.timestamp=}")

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
            ],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)

def test_interrupt_stimplan():
    """
    We do the same test as test_interrupt() except with a StimPlan.
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
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

        burst_channels     : list[int] = [8, 9, 10, 11]
        burst_freqs        : list[int] = [40, 80, 120, 160]
        inter_burst_frames : list[int] = [
            int(1 / freq * frames_per_second)
            for freq in burst_freqs
        ]

        #
        # We build the operations as a list of stim_plans for each tick iteration
        #
        stim_plans: list[StimPlan] = []
        for iteration in range(stop_after_ticks):
            stim_plan: StimPlan = neurons.create_stim_plan()
            stim_plan.interrupt(ChannelSet(*burst_channels))
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
            print(f"{now=} {tick.timestamp=} {now-tick.timestamp=}")

            stim_plans[tick.iteration].run()

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel} {stim.timestamp-tick.timestamp=}")

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
            ],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)

def test_stimplan_at_timestamp():
    """
    We test neurons.interrupt() by:
    - Running a neurons loop with tick frequency of 40 Hz
    - We create two stim plans:
        1. Containing two stims as a burst that can be detected at next tick;
        2. Containing two stims on two different channels separated by an
           interrupt for the first channel. In the next tick, we should only
           detect a single stim on the second channel.
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
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
        stop_after_ticks  = 3

        #
        # We deliver stim bursts based on the burst_channels and burst_freqs
        # so both of these must have the same length. The position relates to
        # the associated tick.iteration.
        #

        burst_channels:     list[list[int]] = [[8, 9], [8, -1, 9]]
        burst_freqs:        list[int]       = [40, 40]
        run_ts_offsets:     list[int]       = [10, 30]

        #
        # We build the operations as a list of stim_plans for each tick iteration
        #

        stim_plans: list[StimPlan] = []
        for tick_iteration in range(len(burst_channels)):
            stim_plan: StimPlan = neurons.create_stim_plan()
            channels            = burst_channels[tick_iteration]
            burst_hz            = burst_freqs[tick_iteration]
            for channel in channels:
                if channel >= 0:
                    stim_plan.stim(
                        ChannelSet(channel),
                        StimDesign(pulse_width_us, -1.0, pulse_width_us, 1.0),
                        BurstDesign(1, burst_hz),
                        )
                else:
                    stim_plan.interrupt(ChannelSet(*range(64)))
            stim_plans.append(stim_plan)

        # We use run_at_timestamp to schedule the stim plans, so that we
        # don't need to run these explicitly during the tick loop.
        for tick_iteration, (run_ts_offset, stim_plan) in enumerate(zip(run_ts_offsets, stim_plans)):
            run_ts = start_timestamp + (tick_frames * tick_iteration) + run_ts_offset
            stim_plan.run_at_timestamp(run_ts)

        observed_tick_stims: dict[int, list[Stim]] = {}
        neurons_loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        for tick in neurons_loop:

            now = neurons.timestamp()
            print(f"{now=} {tick.timestamp=} {now-tick.timestamp=}")

            observed_tick_stims[tick.iteration] = tick.analysis.stims.copy()

            for stim in tick.analysis.stims:
                print(f"\t{stim.timestamp=} {stim.channel} {stim.timestamp-tick.timestamp=}")

        # We should expect to see two stims in iteration 1 and one stim in iteration 2
        expected_tick_stims: dict[int, list[Stim]] = {
            0: [
                Stim(
                    channel=burst_channels[0][0],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 0)
                        + run_ts_offsets[0]
                        + lead_time_frames
                        )
                    ),
                Stim(
                    channel=burst_channels[0][1],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 0)
                        + run_ts_offsets[0]
                        + lead_time_frames
                        )
                    ),
            ],
            1: [
                Stim(
                    channel=burst_channels[1][2],
                    timestamp=(
                        start_timestamp
                        + (tick_frames * 1)
                        + run_ts_offsets[1]
                        + lead_time_frames
                        )
                    ),
            ],
            2: [],
        }

        assert expected_tick_stims == snapshot(observed_tick_stims)
