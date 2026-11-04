"""Batch unit-row Mellum projections using verified original hipBLASLt kernels."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import aiter
import torch

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _solutions():
    config = json.loads(
        Path(os.environ["ATOM_MELLUM_NATIVE_LINEAR_CONFIG"]).read_text()
    )
    if config["torch_version"] != torch.__version__:
        raise RuntimeError("Native Linear profile requires its validated PyTorch build")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.split(":")[0] != config["architecture"]:
        raise RuntimeError(
            "Native Linear profile requires its validated GPU architecture"
        )
    aiter.hipb_create_extension()
    result = {}
    for shape, row in config["solutions"].items():
        actual = aiter.getHipblasltKernelName(row["solution"])
        if actual != row["kernel"]:
            raise RuntimeError(
                "hipBLASLt algorithm index changed; revalidate native Linear profile"
            )
        result[tuple(map(int, shape.split(",")))] = row["solution"]
    LOG.warning("Mellum native hipBLASLt decode solutions enabled: %s", result)
    return result


def try_linear(x, weight, bias=None):
    if (
        bias is not None
        or x.ndim != 2
        or not 1 <= x.shape[0] <= 64
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        return None
    solution = _solutions().get(tuple(weight.shape))
    if solution is None:
        return None
    return aiter.hipb_mm(x, weight.t(), solution)
