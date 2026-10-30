# Anima LoRA training entry point. The trainer lives in library/anima/trainer/;
# this shim only sets the process-level env that must precede the torch import.

import sys

# Windows: suppress per-kernel ptxas.exe/cl.exe console flashes. Must run
# before any subprocess.Popen call (before torch import on Windows).
from library.runtime.proc import install_no_window_default

install_no_window_default()

# Must land before torch initializes the CUDA caching allocator: free-fit
# varies seq_len per step and fragments the pool without expandable segments
# (issue #58). Opt out: ANIMA_EXPANDABLE_SEGMENTS=0.
from library.runtime.allocator import default_expandable_segments

if default_expandable_segments():
    print(
        "Anima: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "(default; set ANIMA_EXPANDABLE_SEGMENTS=0 to disable)"
    )

from library.anima.trainer import (  # noqa: F401  (re-exported for `import train`)
    AnimaTrainer,
    build_network_extras,
    main,
    resolve_network_kwargs,
    setup_parser,
)
from library.config.cli_args import verify_command_line_training_args  # noqa: F401
from library.config.io import read_config_from_file  # noqa: F401
from library.log import setup_logging

setup_logging()

if __name__ == "__main__":
    main(sys.argv[1:])
