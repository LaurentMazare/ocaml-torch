type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

module Linear : sig
  type t

  (** [create output_dim] creates a linear layer with output size
      [output_dim]. *)
  val create : int -> t

  val apply
    :  ?activation:activation (* default: no activation *)
    -> ?use_bias:bool (* default: true *)
    -> t
    -> Tensor.t
    -> Tensor.t

  val vars : t -> Tensor.t list
end
