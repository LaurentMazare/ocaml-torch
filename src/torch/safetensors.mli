open! Base

val read : ?only:string list -> string -> (string * Torch_core.Wrapper.Tensor.t) list
