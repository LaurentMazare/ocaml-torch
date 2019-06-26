open Torch

val load_image : string -> Tensor.t_f32
val load_image_no_resize_and_crop : string -> Tensor.t_f32
val load_images : dir:string -> Tensor.t_f32
val clamp_ : Tensor.t_f32 -> Tensor.t_f32

val load_dataset
  :  dir:string
  -> classes:string list
  -> ?with_cache:string
  -> unit
  -> ('a, 'a) Dataset_helper.t

val write_image : Tensor.t_f32 -> filename:string -> unit

module Classes : sig
  val count : int
  val names : string array
  val top : Tensor.t_f32 -> k:int -> (string * float) list
end

module Loader : sig
  type t

  (* [resize] to 224x224 by default. *)
  val create : ?resize:int * int -> dir:string -> unit -> t
  val random_batch : t -> batch_size:int -> Tensor.t_f32
  val reset : t -> unit
  val next_batch : t -> batch_size:int -> Tensor.t_f32 option
end
