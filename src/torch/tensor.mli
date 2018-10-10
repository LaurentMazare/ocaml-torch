open Torch_core

(* TODO: proper types for Tensor1D, Tensor2D, Tensor3D, ... ? *)
(* TODO: GADT for array element types ? *)
type t = Wrapper_generated.C.Tensor.t

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
val zero_grad : t -> unit

val (+) : t -> t -> t
val (-) : t -> t -> t
val ( * ) : t -> t -> t
val (/) : t -> t -> t
val (+=) : t -> t -> unit
val (-=) : t -> t -> unit
val ( *=) : t -> t -> unit
val (/=) : t -> t -> unit
val (~-) : t -> t
val (=) : t -> t -> t

val mm : t -> t -> t
val f : float -> t
val zeros : ?requires_grad:bool -> ?kind:Kind.t -> ?scale:float -> int list -> t
val ones : ?requires_grad:bool -> ?kind:Kind.t -> ?scale:float -> int list -> t
val rand : ?requires_grad:bool -> ?kind:Kind.t -> ?scale:float -> int list -> t
val randn : ?requires_grad:bool -> ?kind:Kind.t -> ?scale:float -> int list -> t

val shape : t -> int list
val load : string -> t
val save : t -> string -> unit

val backward : t -> unit
val log : t -> t
val softmax : t -> t
val mean : t -> t
val grad : t -> t
val sum : t -> t
val argmax : t -> t
val float_value : t -> float

val fill_float : t -> float -> unit
val get : t -> int -> t
val print : t -> unit
val reshape : t -> dims:int list -> t
val float_vec : ?kind:[< `double | `float | `half > `float ] -> float list -> t
val set_requires_grad : t -> b:bool -> t
val to_type : t -> type_:Kind.t -> t
val to_device : t -> device:Device.t -> t

val narrow : t -> dim:int -> start:int -> len:int -> t

val relu : t -> t
val tanh : t -> t
val sigmoid : t -> t
val leaky_relu : t -> t
val conv2d
  :  ?padding:int*int
  -> ?dilation:int*int
  -> ?groups:int
  -> t (* input *)
  -> t (* weight *)
  -> t (* bias *)
  -> stride:int*int
  -> t

val max_pool2d
  :  ?padding:int*int
  -> ?dilation:int*int
  -> ?ceil_mode:bool
  -> ?stride:int*int
  -> t
  -> ksize:int*int
  -> t
