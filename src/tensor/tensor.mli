(* TODO: proper types for Tensor1D, Tensor2D, Tensor3D, ... ? *)
(* TODO: GADT for array element types ? *)
include module type of Torch_core.Wrapper.Tensor

val set_float2 : t -> int -> int -> float -> unit
val set_float1 : t -> int -> float -> unit
val set_int2 : t -> int -> int -> int -> unit
val set_int1 : t -> int -> int -> unit

val get_float2 : t -> int -> int -> float
val get_float1 : t -> int -> float
val get_int2 : t -> int -> int -> int
val get_int1 : t -> int -> int
