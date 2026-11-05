"""TensorRT engine loading and construction helpers for V2.2 inference."""

from pathlib import Path

import torch


class TensorRTEngine:
    def __init__(self, engine_path, device):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is not installed. Rebuild the project image with the "
                "TensorRT dependencies in docker/Dockerfile."
            ) from exc

        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires an available CUDA GPU.")

        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.stream = torch.cuda.Stream(device=self.device)
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_data = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT failed to create an execution context.")

        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 2 or len(self.output_names) != 1:
            raise RuntimeError(
                "Expected an engine with two inputs and one output; found "
                f"{len(self.input_names)} inputs and {len(self.output_names)} outputs."
            )
        if set(self.input_names) != {"rgb", "modal_x"}:
            raise RuntimeError(
                f"Unexpected TensorRT input names: {self.input_names}."
            )
        self.output_name = self.output_names[0]
        self.input_shapes = {
            name: tuple(self.engine.get_tensor_shape(name))
            for name in self.input_names
        }

    def __call__(self, rgb, modal_x):
        inputs = {"rgb": rgb, "modal_x": modal_x}
        contiguous_inputs = {}
        for name in self.input_names:
            tensor = inputs[name]
            expected_shape = self.input_shapes[name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Input {name} expects {expected_shape}, got {tuple(tensor.shape)}."
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"Input {name} is on {tensor.device}, engine is on {self.device}."
                )
            contiguous_inputs[name] = tensor.contiguous()
            self.context.set_tensor_address(
                name, contiguous_inputs[name].data_ptr()
            )

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        trt_dtype = self.engine.get_tensor_dtype(self.output_name)
        if trt_dtype == self.trt.float32:
            output_dtype = torch.float32
        elif trt_dtype == self.trt.float16:
            output_dtype = torch.float16
        else:
            output_dtype = None
        if output_dtype is None:
            raise RuntimeError(f"Unsupported TensorRT output type: {trt_dtype}")

        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            output = torch.empty(
                output_shape, dtype=output_dtype, device=self.device
            )
            self.context.set_tensor_address(self.output_name, output.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")
        current_stream.wait_stream(self.stream)
        return output


def build_engine_from_onnx(onnx_path, engine_path, precision, workspace_gib):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not installed. Rebuild the project image with the "
            "TensorRT dependencies in docker/Dockerfile."
        ) from exc

    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network()
    parser = trt.OnnxParser(network, builder.logger)
    onnx_data = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_data):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the ONNX model:\n{errors}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1024 ** 3))
    )
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast FP16 support.")
        build_config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, build_config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build the engine.")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    return engine_path
