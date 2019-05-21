open Base
open Torch

(** [load_image ?resize filename] returns a tensor containing the pixels for the image
    in [filename]. Supported image formats are JPEG and PNG.
    The resulting tensor has dimensions NCHW (with N = 1) with values between 0 and 255.
    When [resize] is set, the image is first resized preserving its original ratio
    then a center crop is taken.
*)
val load_image : ?resize:int * int -> string -> Tensor.t Or_error.t

(** [load_images ?resize dir_name] is similar to applying [load_image] to all the images
    in [dir_name].
    The resulting tensor has dimensions NCHW where N is the number of images.
*)
val load_images : ?resize:int * int -> string -> Tensor.t

(** [load_dataset ~dir ~classes ~with_cache ~resize] loads the images contained in
    directories [dir/class] where class ranges over [classes]. The class is used
    to determine the labels in the resulting dataset.
    [resize] should be used if the images don't have all the same size.
*)
val load_dataset :
     dir:string
  -> classes:string list
  -> with_cache:string option
  -> resize:int * int
  -> Dataset_helper.t

(** [write_image tensor ~filename] writes [tensor] as an image to the disk.
    The format is determined by [filename]'s extension, defaulting to png.
    Supported formats are [jpg], [tga], [bmp], and [png].
    The tensor values should be between 0 and 255, the shape of the tensor
    can be [1; channels; height; width] or [channels; height; width]
    where channels is either 1 or 3.
*)
val write_image : Tensor.t -> filename:string -> unit

module Loader : sig
  type t

  val create : ?resize:int * int -> dir:string -> unit -> t
  val random_batch : t -> batch_size:int -> Tensor.t
  val reset : t -> unit
  val next_batch : t -> batch_size:int -> Tensor.t option
end

(** [resize t ~height ~width] resizes the given tensor to [height] and [width].
    This does not preserve the aspect ratio.
    [t] can have dimensions NCHW with C set to 1 or 3, the returned tensor will
    have dimensions NCH'W' with [H' = height] and [W' = width].
    The input and output tensors have values between 0 and 255.
*)
val resize : Tensor.t -> height:int -> width:int -> Tensor.t
