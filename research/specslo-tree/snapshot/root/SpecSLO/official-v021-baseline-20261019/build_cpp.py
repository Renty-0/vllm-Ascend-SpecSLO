"""Build the complete unchanged official C++ extension with its CMake API."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pybind11
import torch_npu

ROOT = Path('/root/SpecSLO/official-v021-baseline-20261019')
SOURCE = ROOT / 'vllm-ascend'
BUILD = ROOT / 'cpp-build'
arguments = ['cmake', '-S', str(SOURCE), '-B', str(BUILD), '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_INSTALL_PREFIX=' + str(SOURCE / 'vllm_ascend'),
             '-DASCEND_HOME_PATH=' + os.environ['ASCEND_HOME_PATH'],
             '-DPYTHON_EXECUTABLE=' + sys.executable,
             '-DPYTHON_INCLUDE_PATH=' + sysconfig.get_paths()['include'],
             '-Dpybind11_DIR=' + pybind11.get_cmake_dir(),
             '-DTORCH_NPU_PATH=' + str(Path(torch_npu.__file__).parent),
             '-DSOC_VERSION=ascend910b1']
commands = [arguments, ['cmake', '--build', str(BUILD), '--parallel', '4'],
            ['cmake', '--install', str(BUILD)]]
(ROOT / 'CPP_BUILD_COMMANDS.json').write_text(json.dumps(commands, indent=2) + '\n')
for command in commands:
    subprocess.run(command, check=True)
