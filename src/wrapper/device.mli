type t =
  | Cpu
  | Cuda

val to_int : t -> int
