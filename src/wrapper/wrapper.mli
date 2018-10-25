module Scalar : sig
  type t
  val int : int -> t
  val float : float -> t
end

module Tensor : sig
  type t
  include Wrapper_generated_intf.S with type t := t and type scalar := Scalar.t

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

  val reshape : t -> dims:int list -> t
  val shape : t -> int list
  val kind : t -> Kind.t
  val requires_grad : t -> bool
  val get : t -> int -> t
  val select : t -> dim:int -> index:int -> t

  val float_value : t -> float
  val int_value : t -> int
  val fill_float : t -> float -> unit
  val fill_int : t -> int -> unit

  val backward : t -> unit

  val print : t -> unit

  val sum : t -> t
  val mean : t -> t
  val argmax : t -> t
  val softmax : t -> t
  val nll_loss_ : ?reduction:Reduction.t -> t -> targets:t -> t

  val defined : t -> bool

  val copy_ : t -> src:t -> unit
end

module Optimizer : sig
  type t

  val adam : Tensor.t list -> learning_rate:float -> t

  val sgd
    :  Tensor.t list
    -> learning_rate:float
    -> momentum:float
    -> dampening:float
    -> weight_decay:float
    -> nesterov:bool
    -> t

  val set_learning_rate
    :  t
    -> float
    -> unit

  val zero_grad : t -> unit
  val step : t -> unit
end

module Serialize : sig
  val save : Tensor.t -> filename:string -> unit
  val load : filename:string -> Tensor.t
  val save_multi : named_tensors:(string * Tensor.t) list -> filename:string -> unit
  val load_multi : names:string list -> filename:string -> Tensor.t list
  val load_multi_ : named_tensors:(string * Tensor.t) list -> filename:string -> unit
end

module Cuda : sig
  val device_count : unit -> int
  val is_available : unit -> bool
  val cudnn_is_available : unit -> bool
end
