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

(* [no_grad t ~f] runs [f] on [t] without tracking gradients for t. *)
val no_grad : t -> f:(t -> 'a) -> 'a

val (+) : t -> t -> t
val (-) : t -> t -> t
val ( * ) : t -> t -> t
val (/) : t -> t -> t
val (-=) : t -> t -> unit
val (~-) : t -> t
val (=) : t -> t -> t

val mm : t -> t -> t
val f : float -> t
val zeros : ?requires_grad:bool -> ?kind:Torch_core.Wrapper.Kind.t -> int list -> t
