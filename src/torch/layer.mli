module Var_store : sig
  type t
  val create : unit -> t
  val vars : t -> Tensor.t list
end

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

module Linear : sig
  type t

  val create : Var_store.t -> input_dim:int -> int -> t

  val apply
    :  ?activation:activation (* default: no activation *)
    -> ?use_bias:bool (* default: true *)
    -> t
    -> Tensor.t
    -> Tensor.t
end

module Conv2D : sig
  type t

  val create
    :  Var_store.t
    -> ksize:int * int
    -> stride:int * int
    -> ?padding:int * int
    -> input_dim:int
    -> int
    -> t

  val create_
    :  Var_store.t
    -> ksize:int
    -> stride:int
    -> ?padding:int
    -> input_dim:int
    -> int
    -> t

  val apply
    :  ?activation:activation (* default: no activation *)
    -> t
    -> Tensor.t
    -> Tensor.t
end

module ConvTranspose2D : sig
  type t

  val create
    :  Var_store.t
    -> ksize:int * int
    -> stride:int * int
    -> ?padding:int * int
    -> ?output_padding:int * int
    -> input_dim:int
    -> int
    -> t

  val create_
    :  Var_store.t
    -> ksize:int
    -> stride:int
    -> ?padding:int
    -> ?output_padding:int
    -> input_dim:int
    -> int
    -> t

  val apply
    :  ?activation:activation (* default: no activation *)
    -> t
    -> Tensor.t
    -> Tensor.t
end
