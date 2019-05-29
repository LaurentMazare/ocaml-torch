type t = Torch_core.Device.t =
  | Cpu
  | Cuda of int

val cuda_if_available : unit -> t
