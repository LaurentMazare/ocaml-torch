val manual_seed : int -> unit
module Scalar : sig
  type t
  val int : int -> t
  val float : float -> t
end

module Tensor : sig
  type t
  include Wrapper_generated_intf.S with type t := t and type scalar := Scalar.t

  val new_tensor : unit -> t

  val float_vec
    :  ?kind:[ `double | `float | `half ]
    -> float list
    -> t

  val int_vec
    :  ?kind:[ `int | `int16 | `int64 | `int8 | `uint8 ]
    -> int list
    -> t

  val of_bigarray : (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> t
  val copy_to_bigarray : t -> (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit

  val shape : t -> int list
  val shape1_exn : t -> int
  val shape2_exn : t -> int * int
  val shape3_exn : t -> int * int * int
  val shape4_exn : t -> int * int * int * int
  val kind : t -> Kind.t
  val requires_grad : t -> bool
  val grad_set_enabled : bool -> bool (* returns the previous state. *)
  val get : t -> int -> t
  val select : t -> dim:int -> index:int -> t

  val float_value : t -> float
  val int_value : t -> int
  val float_get : t -> int list -> float
  val int_get : t -> int list -> int
  val float_set : t -> int list -> float -> unit
  val int_set : t -> int list -> int -> unit
  val fill_float : t -> float -> unit
  val fill_int : t -> int -> unit

  val backward : ?keep_graph:bool -> ?create_graph:bool -> t -> unit

  (* Computes and returns the sum of gradients of outputs w.r.t. the inputs.
     If [create_graph] is set to true, graph of the derivative will be constructed,
     allowing to compute higher order derivative products.
  *)
  val run_backward
    :  ?keep_graph:bool
    -> ?create_graph:bool
    -> t list
    -> t list
    -> t list

  val print : t -> unit
  val to_string : t -> line_size:int -> string

  val sum : t -> t
  val mean : t -> t
  val argmax : t -> t

  val defined : t -> bool

  val copy_ : t -> src:t -> unit
end

module Optimizer : sig
  type t

  val adam
    :  learning_rate:float
    -> beta1:float
    -> beta2:float
    -> weight_decay:float
    -> t

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

  val add_parameters : t -> Tensor.t list -> unit
  val zero_grad : t -> unit
  val step : t -> unit
end

module Serialize : sig
  val save : Tensor.t -> filename:string -> unit
  val load : filename:string -> Tensor.t
  val save_multi : named_tensors:(string * Tensor.t) list -> filename:string -> unit
  val load_multi : names:string list -> filename:string -> Tensor.t list
  val load_multi_ : named_tensors:(string * Tensor.t) list -> filename:string -> unit
  val load_all : filename:string -> (string * Tensor.t) list
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
  val tensor : Tensor.t -> t
  val int64 : Int64.t -> t
  val double : float -> t
  val tuple : t list -> t
  val tag : t -> Tag.t

  val to_tensor : t -> Tensor.t
  val to_int64 : t -> Int64.t
  val to_double : t -> float
  val to_tuple : t -> t list
end

module Module : sig
  type t
  val load : string -> t
  val forward : t -> Tensor.t list -> Tensor.t
  val forward_ : t -> Ivalue.t list -> Ivalue.t
end
