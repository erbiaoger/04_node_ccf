"""Strict native boundary; explicit CPU/Torch references have no fallback."""

from __future__ import annotations

import numpy as np
from dasqt.features.dispersion.backend.components.backends.cpp_cuda_backend import (
    require_native,
)

from .parameters import VERSION
from .reference import ReferenceSession


class NativeSession:
    def __init__(self, params, stations):
        module = require_native()
        if "file_stacking_method" in params and not getattr(
            module, "node_independent_minute_stack", False
        ):
            raise RuntimeError(
                "Rebuild native/cc_cuda_engine for independent minute/output stacking"
            )
        if getattr(module, "node_pairwise_version", None) != VERSION:
            raise RuntimeError(
                "Rebuild native/cc_cuda_engine: extension lacks node spectral pair PWS v1"
            )
        if params.get("cc_upload_block", 1) and not getattr(
            module, "node_block_upload", False
        ):
            raise RuntimeError(
                "Rebuild native/cc_cuda_engine for complete 30-minute block uploads"
            )
        if not getattr(module, "node_cross_minute_batch", False):
            raise RuntimeError(
                "Rebuild native/cc_cuda_engine for cross-minute CUDA batches"
            )
        if params.get("cc_reuse_workspace", 1) and not getattr(
            module, "node_workspace_reuse", False
        ):
            raise RuntimeError(
                "Rebuild native/cc_cuda_engine for reusable CUDA workspaces"
            )
        self.native = module.NodePairSession(params, stations)

    def add_block(self, data, valid, starts, minute_ids):
        self.native.add_block(
            np.ascontiguousarray(data, dtype=np.float32),
            np.ascontiguousarray(valid, dtype=np.uint8),
            np.asarray(starts, dtype=np.int64).tolist(),
            np.asarray(minute_ids, dtype=np.int64).tolist(),
        )

    def stack_pairs(self, data, valid):
        self.native.stack_pairs(
            np.ascontiguousarray(data, dtype=np.float32),
            np.ascontiguousarray(valid, dtype=np.uint8),
        )

    def finish_group(self):
        return dict(self.native.finish_group())

    def clear_group(self):
        self.native.clear_group()

    def diagnostics(self):
        return dict(self.native.diagnostics())

    def cancel(self):
        self.native.cancel()


def create_session(params, stations):
    return (
        NativeSession(params, stations)
        if params["cc_backend"] == "cpp_cuda"
        else ReferenceSession(params, stations)
    )
