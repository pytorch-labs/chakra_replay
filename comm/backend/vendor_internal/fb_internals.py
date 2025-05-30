import logging


logger = logging.getLogger(__name__)
logger.info("importing FB internals.")

try:
    # import pytorch_mtia_backend to register mtia backend to use mtia/hccl
    import param_bench.et_replay.comm.backend.vendor_internal.pytorch_mtia_backend  # noqa
except ImportError as e:
    print(f"ImportError: {e}")
    logger.info(
        "pytorch_mtia_backend not found. Build PARAM Bench with '-c fbcode.use_hccl=True' to enable it."
    )
    pass
