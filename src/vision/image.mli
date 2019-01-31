open Base
open Torch

(** [load_image ?resize filename] returns a tensor containing the pixels for the image
    in [filename]. Supported image formats are JPEG and PNG.
    The resulting tensor has dimensions NCHW (with N = 1).
    When [resize] is set, the image is first resized preserving its original ratio
    then a center crop is taken.
*)
val load_image : ?resize:int * int -> string -> Tensor.t Or_error.t

(** [load_images ?resize dir_name] is similar to applying [load_image] to all the images
    in [dir_name].
    The resulting tensor has dimensions NCHW where N is the number of images.
*)
val load_images : ?resize:int * int -> string -> Tensor.t

val load_dataset :
     dir:string
  -> classes:string list
  -> with_cache:string option
  -> resize:int * int
  -> Dataset_helper.t

val write_image : Tensor.t -> filename:string -> unit

module Loader : sig
  type t

  val create : ?resize:int * int -> dir:string -> unit -> t
  val random_batch : t -> batch_size:int -> Tensor.t
  val reset : t -> unit
  val next_batch : t -> batch_size:int -> Tensor.t option
end
