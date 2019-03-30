type t = Torch_core.Device.t =
  | Cpu
  | Cuda

val cuda_if_available : unit -> t
