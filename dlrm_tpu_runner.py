import sys
import argparse

from torch_xla.experimental import pjrt

from dlrm_s_pytorch import main, parse_args


if __name__ == '__main__':
    pre_spawn_parser = argparse.ArgumentParser()
    pre_spawn_parser.add_argument("--tpu-cores", type=int, default=8)
    pre_spawn_flags, _ = pre_spawn_parser.parse_known_args()
    pjrt.run_multiprocess(main)
    # xmp.spawn(main, args=(), nprocs=pre_spawn_flags.tpu_cores)
