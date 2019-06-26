(* An annotated dataset for images. *)
type ('a, 'b) t =
  { train_images : 'a Tensor.t
  ; train_labels : 'b Tensor.t
  ; test_images : 'a Tensor.t
  ; test_labels : 'b Tensor.t
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
  :  ?device:Device.t
  -> ?augmentation:[ `flip | `crop_with_pad of int | `cutout of int ] list
  -> ('a, 'b) t
  -> batch_size:int
  -> batch_idx:int
  -> 'a Tensor.t * 'b Tensor.t

(** [batch_accuracy ?device ?samples t test_or_train ~batch_size ~predict] computes
    the accuracy of applying [predict] to test or train images as specified by
    [test_or_train].
    Computations are done using batch of length [batch_size].
*)
val batch_accuracy
  :  ?device:Device.t
  -> ?samples:int
  -> ('a, 'b) t
  -> [ `test | `train ]
  -> batch_size:int
  -> predict:('a Tensor.t -> 'b Tensor.t)
  -> float

(** [read_with_cache ~cache_file ~read] either returns the content of [cache_file] if
    present or regenerate the file using [read] if not.
*)
val read_with_cache : cache_file:string -> read:(unit -> ('a, 'b) t) -> ('a, 'b) t

(** [batches_per_epoch t ~batch_size] returns the total number of batches of size
    [batch_size] in [t]. *)
val batches_per_epoch : (_, _) t -> batch_size:int -> int

(** [iter ?device ?augmentation ?shuffle t ~f ~batch_size] iterates function [f] on
    all the batches from [t] taken with a size [batch_size].
    Random shuffling and augmentation can be specified.
*)
val iter
  :  ?device:Device.t
  -> ?augmentation:[ `flip | `crop_with_pad of int | `cutout of int ] list
  -> ?shuffle:bool
  -> ('a, 'b) t
  -> f:(int -> batch_images:'a Tensor.t -> batch_labels:'b Tensor.t -> unit)
  -> batch_size:int
  -> unit

(** [random_flip t] applies some random flips to a tensor of dimension [ N; H; C; W].
    The last dimension related to width can be flipped.
*)
val random_flip : 'a Tensor.t -> 'a Tensor.t

(** [random_crop t ~pad] performs some data augmentation by padding [t] with zeros on
    the two last dimensions with [pad] new values on each side, then performs some
    random crop to go back to the original shape.
*)
val random_crop : 'a Tensor.t -> pad:int -> 'a Tensor.t

(** [shuffle t] returns [t] where training images and labels have been shuffled. *)
val shuffle : ('a, 'b) t -> ('a, 'b) t

val map
  :  ?device:Device.t
  -> ('a, 'b) t
  -> f:(int
        -> batch_images:'a Tensor.t
        -> batch_labels:'b Tensor.t
        -> 'c Tensor.t * 'd Tensor.t)
  -> batch_size:int
  -> ('c, 'd) t

val print_summary : (_, _) t -> unit

(** [read_char_tensor filename] returns a tensor of char containing the specified
    file.
*)
val read_char_tensor : string -> _ Tensor.t
