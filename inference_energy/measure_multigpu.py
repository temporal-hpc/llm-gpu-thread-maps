#!/usr/bin/env python3
"""GPU power monitor: runs a command and logs the power of all visible GPUs.

Polls nvidia-smi every 100 ms while the command runs, adds the power of
every visible GPU and integrates it over time. The log (power-gpu_log.dat)
has one line per sample:

    timestep power acc-energy avg-power dt acc-time
             [W]   [J]        [W]       [s] [s]

Usage: python3 measure_multigpu.py <command> [args...]
"""
import subprocess
import sys
import time

LOG_FILE = "power-gpu_log.dat"
INTERVAL = 0.1  # seconds


def get_total_gpu_power():
    """Sum of the instantaneous power of all visible GPUs, in watts."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        powers = [float(x) for x in result.strip().split('\n') if x.strip()]
        return sum(powers), len(powers)
    except Exception:
        return 0.0, 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 measure_multigpu.py <command>")
        sys.exit(1)

    command = sys.argv[1:]
    print(f"--- GPU monitor: measuring {command} ---")

    proc = subprocess.Popen(command)
    start_time = time.time()
    total_energy_joules = 0.0
    last_check = start_time

    try:
        with open(LOG_FILE, 'w') as f:
            f.write("timestep power acc-energy avg-power dt acc-time\n")
            step = 0
            while proc.poll() is None:
                current_time = time.time()
                dt = current_time - last_check

                total_watts, num_gpus = get_total_gpu_power()

                # energy of this interval (joules = watts * seconds)
                energy_step = total_watts * dt
                total_energy_joules += energy_step

                elapsed_total = current_time - start_time
                avg_power = total_energy_joules / elapsed_total if elapsed_total > 0 else total_watts

                f.write(f"{step} {total_watts:.2f} {total_energy_joules:.4f} {avg_power:.2f} {dt:.4f} {elapsed_total:.4f}\n")
                f.flush()

                step += 1
                last_check = current_time

                sleep_time = INTERVAL - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()
    print(f"--- GPU monitor done. Total energy: {total_energy_joules:.2f} J ---")


if __name__ == "__main__":
    main()
