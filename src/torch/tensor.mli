open Torch_core

(* TODO: proper types for Tensor1D, Tensor2D, Tensor3D, ... ? *)
(* TODO: GADT for array element types ? *)
type t = Torch_core.Wrapper.Tensor.t
include module type of Torch_core.Wrapper.Tensor with type t := t

val set_float2 : t -> int -> int -> float -> unit
val set_float1 : t -> int -> float -> unit
val set_int2 : t -> int -> int -> int -> unit
val set_int1 : t -> int -> int -> unit

val get_float2 : t -> int -> int -> float
val get_float1 : t -> int -> float
val get_int2 : t -> int -> int -> int
val get_int1 : t -> int -> int

val ( .%{} ) : t -> int list -> int
val ( .%{}<- ) : t -> int list -> int -> unit
val ( .%.{} ) : t -> int list -> float
val ( .%.{}<- ) : t -> int list -> float -> unit

val ( .%[] ) : t -> int -> int
val ( .%[]<- ) : t -> int -> int -> unit
val ( .%.[] ) : t -> int -> float
val ( .%.[]<- ) : t -> int -> float -> unit

(* [no_grad_ t ~f] runs [f] on [t] without tracking gradients for t. *)
val no_grad_ : t -> f:(t -> 'a) -> 'a
val no_grad : (unit -> 'a) -> 'a
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

val float_vec
  :  ?kind:[ `double | `float | `half ]
  -> ?device:Torch_core.Device.t
  -> float list
  -> t

val to_type : t -> type_:Kind.t -> t
val to_device : ?device:Device.t -> t -> t

val to_float0 : t -> float option
val to_float1 : t -> float array option
val to_float2 : t -> float array array option
val to_float3 : t -> float array array array option

val to_float0_exn : t -> float
val to_float1_exn : t -> float array
val to_float2_exn : t -> float array array
val to_float3_exn : t -> float array array array

val to_int0 : t -> int option
val to_int1 : t -> int array option
val to_int2 : t -> int array array option
val to_int3 : t -> int array array array option

val to_int0_exn : t -> int
val to_int1_exn : t -> int array
val to_int2_exn : t -> int array array
val to_int3_exn : t -> int array array array

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

val const_batch_norm : ?momentum:float -> ?eps:float -> t -> t

val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> t
val copy_to_bigarray : t -> ('b, 'a, Bigarray.c_layout) Bigarray.Genarray.t -> unit
val to_bigarray : t -> kind:('a, 'b) Bigarray.kind  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

val cross_entropy_for_logits : ?reduction:Reduction.t -> t -> targets:t -> t

val dropout : t -> p:float (* dropout prob *) -> is_training:bool -> t
val nll_loss
  :  ?reduction:Torch_core.Reduction.t
  -> t
  -> targets:t
  -> t

val bce_loss
  :  ?reduction:Torch_core.Reduction.t
  -> t
  -> targets:t
  -> t

val mse_loss
  :  ?reduction:Torch_core.Reduction.t
  -> t
  -> t
  -> t

val undefined : t Lazy.t

val pp : Format.formatter -> t -> unit
val copy : t -> t
val print_shape : ?name:string -> t -> unit
