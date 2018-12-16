open Base
open Torch

val load_image
  :  ?resize:(int * int)
  -> string
  -> Tensor.t Or_error.t

val load_images : ?resize:(int * int) -> string -> Tensor.t

val load_dataset
  :  dir:string
  -> classes:string list
  -> with_cache:string option
  -> resize:(int * int)
  -> Dataset_helper.t

val write_image : Tensor.t -> filename:string -> unit
