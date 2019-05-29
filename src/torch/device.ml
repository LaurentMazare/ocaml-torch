type t = Torch_core.Device.t =
  | Cpu
  | Cuda of int

let cuda_if_available () = if Cuda.is_available () then Cuda 0 else Cpu
