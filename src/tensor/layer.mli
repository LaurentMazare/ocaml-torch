type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

module Linear : sig
  type t

  val create : input_dim:int -> int -> t

  val apply
    :  ?activation:activation (* default: no activation *)
    -> ?use_bias:bool (* default: true *)
    -> t
    -> Tensor.t
    -> Tensor.t

  val vars : t -> Tensor.t list
end
