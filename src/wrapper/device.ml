type t =
  | Cpu
  | Cuda

(* Hardcoded, should match Device.h *)
let to_int = function Cpu -> 0 | Cuda -> 1
