type t = Torch_core.Device.t =
  | Cpu
  | Cuda

let cuda_if_available () = if Cuda.is_available () then Cuda else Cpu
