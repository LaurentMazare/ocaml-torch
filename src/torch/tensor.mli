open Torch_core

(* TODO: proper types for Tensor1D, Tensor2D, Tensor3D, ... ? *)
(* TODO: GADT for array element types ? *)
type t
val to_ptr : t -> Wrapper_generated.C.Tensor.t
val of_ptr : Wrapper_generated.C.Tensor.t -> t

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

type create
  =  ?requires_grad:bool
  -> ?kind:Torch_core.Kind.t
  -> ?device:Torch_core.Device.t
  -> ?scale:float
  -> int list
  -> t

val zeros : create
val ones : create
val rand : create
val randn : create

val defined : t -> bool
val shape : t -> int list

val backward : t -> unit
val log : t -> t
val softmax : t -> t
val log_softmax : t -> t
val mean : t -> t
val grad : t -> t
val sum : t -> t
val argmax : t -> t
val float_value : t -> float

val fill_float : t -> float -> unit
val get : t -> int -> t
val print : t -> unit
val reshape : t -> dims:int list -> t
val float_vec
  :  ?kind:[< `double | `float | `half > `float ]
  -> ?device:Torch_core.Device.t
  -> float list
  -> t

val set_requires_grad : t -> b:bool -> t
val to_type : t -> type_:Kind.t -> t
val to_device : ?device:Device.t -> t -> t

val narrow : t -> dim:int -> start:int -> len:int -> t

val relu : t -> t
val tanh : t -> t
val sigmoid : t -> t
val leaky_relu : t -> t

val cat : t list -> dim:int -> t

val conv2d
  :  ?padding:int*int
  -> ?dilation:int*int
  -> ?groups:int
  -> t (* input *)
  -> t (* weight *)
  -> t (* bias *)
  -> stride:int*int
  -> t

val conv_transpose2d
  :  ?output_padding:int*int
  -> ?padding:int*int
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

val avg_pool2d
  :  ?padding:int*int
  -> ?count_include_pad:bool
  -> ?ceil_mode:bool
  -> ?stride:int*int
  -> t
  -> ksize:int*int
  -> t

val batch_norm
  :  t
  -> t option
  -> t option
  -> t option
  -> t option
  -> bool
  -> float
  -> float
  -> bool
  -> t

val dropout : t -> p:float (* dropout probability *) -> is_training:bool -> t

val const_batch_norm : ?momentum:float -> ?eps:float -> t -> t

val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> t
val copy_to_bigarray : t -> ('b, 'a, Bigarray.c_layout) Bigarray.Genarray.t -> unit
val to_bigarray : t -> kind:('a, 'b) Bigarray.kind  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
