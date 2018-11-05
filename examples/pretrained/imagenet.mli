open Torch

val load_image : string -> Tensor.t

module Classes : sig
  val count : int
  val names : string array
  val top : Tensor.t -> k:int -> (string * float) list
end
