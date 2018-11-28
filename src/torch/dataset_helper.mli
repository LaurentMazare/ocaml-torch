(* An annotated dataset for images. *)
type t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

(** [train_batch ?device ?augmentation t ~batch_size ~batch_idx] returns two tensors
    corresponding to a training batch. The first tensor is for images and the second
    for labels.
    Each batch has a first dimension of size [batch_size], [batch_idx] = 0 returns
    the first batch, [batch_idx] = 1 returns the second one and so on.
    The tensors are located on [device] if provided. Random data augmentation is
    performed as specified via [augmentation].
*)
val train_batch
  :  ?device:Torch_core.Device.t
  -> ?augmentation:[ `flip | `crop_with_pad of int | `cutout of int ] list
  -> t
  -> batch_size:int
  -> batch_idx:int
  -> Tensor.t * Tensor.t

(** [batch_accuracy ?device ?samples t test_or_train ~batch_size ~predict] computes
    the accuracy of applying [predict] to test or train images as specified by
    [test_or_train].
    Computations are done using batch of length [batch_size].
*)
val batch_accuracy
  :  ?device:Torch_core.Device.t
  -> ?samples:int
  -> t
  -> [ `test | `train ]
  -> batch_size:int
  -> predict:(Tensor.t -> Tensor.t)
  -> float

(** [read_with_cache ~cache_file ~read] either returns the content of [cache_file] if
    present or regenerate the file using [read] if not.
*)
val read_with_cache
  :  cache_file:string
  -> read:(unit -> t)
  -> t

(** [batches_per_epoch t ~batch_size] returns the total number of batches of size
    [batch_size] in [t]. *)
val batches_per_epoch : t -> batch_size:int -> int

(** [iter ?device ?augmentation ?shuffle t ~f ~batch_size] iterates function [f] on
    all the batches from [t] taken with a size [batch_size].
    Random shuffling and augmentation can be specified.
*)
val iter
  :  ?device:Torch_core.Device.t
  -> ?augmentation:[ `flip | `crop_with_pad of int | `cutout of int ] list
  -> ?shuffle:bool
  -> t
  -> f:(int -> batch_images:Tensor.t -> batch_labels:Tensor.t -> unit)
  -> batch_size:int
  -> unit

(** [random_flip t] applies some random flips to a tensor of dimension [ N; H; C; W].
    The last dimension related to width can be flipped.
*)
val random_flip : Tensor.t -> Tensor.t

(** [random_crop t ~pad] performs some data augmentation by padding [t] with zeros on
    the two last dimensions with [pad] new values on each side, then performs some
    random crop to go back to the original shape.
*)
val random_crop : Tensor.t -> pad:int -> Tensor.t

(** [shuffle t] returns [t] where training images and labels have been shuffled. *)
val shuffle : t -> t

val map
  :  ?device:Torch_core.Device.t
  -> t
  -> f:(int -> batch_images:Tensor.t -> batch_labels:Tensor.t -> Tensor.t * Tensor.t)
  -> batch_size:int
  -> t

val print_summary : t -> unit

(** [read_char_tensor filename] returns a tensor of char containing the specified
    file.
*)
val read_char_tensor : string -> Tensor.t
