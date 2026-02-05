"""Generated proto files for cuvis.ai gRPC API v1."""

__all__: list[str] = []

try:
    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc

    __all__ = ["cuvis_ai_pb2", "cuvis_ai_pb2_grpc"]
except ImportError:
    # Proto files not generated yet or proto extra not installed
    pass
