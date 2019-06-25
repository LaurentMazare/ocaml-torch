val manual_seed : int -> unit

module Scalar : sig
  type _ t

  val int : int -> int t
  val float : float -> float t
end

module Tensor : sig
  include Wrapper_generated_intf.S with type 'a scalar := 'a Scalar.t
  type packed = T : _ t -> packed

  val extract : packed -> kind:'a Kind.t -> 'a t option

  val new_tensor : unit -> _ t
  val float_vec : ?kind:[ `double | `float | `half ] -> float list -> _ t
  val int_vec : ?kind:[ `int | `int16 | `int64 | `int8 | `uint8 ] -> int list -> _ t
  val of_bigarray : (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> _ t
  val copy_to_bigarray : _ t -> (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit
  val shape : _ t -> int list
  val size : _ t -> int list
  val shape1_exn : _ t -> int
  val shape2_exn : _ t -> int * int
  val shape3_exn : _ t -> int * int * int
  val shape4_exn : _ t -> int * int * int * int
  val kind : _ t -> Kind.packed
  val requires_grad : _ t -> bool
  val grad_set_enabled : bool -> bool

  (* returns the previous state. *)
  val get : 'a t -> int -> 'a t
  val select : 'a t -> dim:int -> index:int -> 'a t
  val float_value : _ t -> float
  val int_value : _ t -> int
  val float_get : _ t -> int list -> float
  val int_get : _ t -> int list -> int
  val float_set : _ t -> int list -> float -> unit
  val int_set : _ t -> int list -> int -> unit
  val fill_float : _ t -> float -> unit
  val fill_int : _ t -> int -> unit
  val backward : ?keep_graph:bool -> ?create_graph:bool -> _ t -> unit

  (* Computes and returns the sum of gradients of outputs w.r.t. the inputs.
     If [create_graph] is set to true, graph of the derivative will be constructed,
     allowing to compute higher order derivative products.
  *)
  val run_backward : ?keep_graph:bool -> ?create_graph:bool -> 'a t list -> 'a t list -> 'a t list
  val print : _ t -> unit
  val to_string : _ t -> line_size:int -> string
  val sum : 'a t -> 'a t
  val mean : 'a t -> 'a t
  val argmax : ?dim:int -> ?keepdim:bool -> 'a t -> 'a t
  val defined : _ t -> bool
  (* TODO: more restrictive types. *)
  val copy_ : 'a t -> src:'b t -> unit
  val max : 'a t -> 'a t -> 'a t
  val min : 'a t -> 'a t -> 'a t
end

module Optimizer : sig
  type t

  val adam : learning_rate:float -> beta1:float -> beta2:float -> weight_decay:float -> t

  val rmsprop
    :  learning_rate:float
    -> alpha:float
    -> eps:float
    -> weight_decay:float
    -> momentum:float
    -> centered:bool
    -> t

  val sgd
    :  learning_rate:float
    -> momentum:float
    -> dampening:float
    -> weight_decay:float
    -> nesterov:bool
    -> t

  val set_learning_rate : t -> float -> unit
  val set_momentum : t -> float -> unit
  val add_parameters : t -> _ Tensor.t list -> unit
  val zero_grad : t -> unit
  val step : t -> unit
end

module Serialize : sig
  val save : _ Tensor.t -> filename:string -> unit
  val load : filename:string -> Tensor.packed
  val save_multi : named_tensors:(string * Tensor.packed) list -> filename:string -> unit
  val load_multi : names:string list -> filename:string -> Tensor.packed list
  val load_multi_ : named_tensors:(string * Tensor.packed) list -> filename:string -> unit
  val load_all : filename:string -> (string * Tensor.packed) list
end

module Cuda : sig
  val device_count : unit -> int
  val is_available : unit -> bool
  val cudnn_is_available : unit -> bool
  val set_benchmark_cudnn : bool -> unit
end

module Ivalue : sig
  module Tag : sig
    type t =
      | Tensor
      | Int
      | Double
      | Tuple
  end

  type t

  val tensor : _ Tensor.t -> t
  val int64 : Int64.t -> t
  val double : float -> t
  val tuple : t list -> t
  val tag : t -> Tag.t
  val to_tensor : t -> Tensor.packed
  val to_int64 : t -> Int64.t
  val to_double : t -> float
  val to_tuple : t -> t list
end

module Module : sig
  type t

  val load : string -> t
  val forward : t -> _ Tensor.t list -> _ Tensor.t
  val forward_ : t -> Ivalue.t list -> Ivalue.t
end
