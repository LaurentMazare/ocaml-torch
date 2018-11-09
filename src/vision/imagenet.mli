open Torch

val load_image : string -> Tensor.t
val load_images : dir:string -> Tensor.t
val load_dataset
  :  dir:string
  -> classes:string list
  -> with_cache:string option
  -> Dataset_helper.t

module Classes : sig
  val count : int
  val names : string array
  val top : Tensor.t -> k:int -> (string * float) list
end
