#%%

import os

os.environ["CL_SDK_WEBSOCKET"] = "1"
os.environ["CL_SDK_ACCELERATED_TIME"] = "0"

import time

import cl

import logging

# logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    with cl.open() as neurons:

        # print("Neurons opened", flush=True)
        # time.sleep(5)

        input("Press Enter to continue...")

        for i in range(10):
            print(f"Iteration {i}")

            print("Stimulating channel 3 with 1.0 uA")
            for _ in range(50):
                neurons.stim(3, 1.0)
                time.sleep(0.1)

            # neurons.stim(3, 1.0, cl.BurstDesign(10, 10))
            # time.sleep(5)

            print("Entering loop for channel 5 stimulation")
            for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=5):
                neurons.stim(5, 1.0)

            print("Entering loop for channel 10 stimulation")
            for tick in neurons.loop(ticks_per_second=200, stop_after_seconds=5):
                neurons.stim(10, 2.0)

            print("Entering loop for channel 5 stimulation")
            for tick in neurons.loop(ticks_per_second=1, stop_after_seconds=5):
                neurons.stim(5, 1.0)

            print("Entering loop for channel 3 stimulation")
            for tick in neurons.loop(ticks_per_second=60, stop_after_seconds=5):
                neurons.stim(3, 1.0)

            print("Entering loop for channel 6 stimulation")
            for tick in neurons.loop(ticks_per_second=500, stop_after_seconds=5):
                neurons.stim(6, 1.0)

# %%
