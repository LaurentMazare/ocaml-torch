type t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

val train_batch
  :  ?device:Torch_core.Device.t
  -> ?augmentation:[ `flip | `crop_with_pad of int | `flip_and_crop_with_pad of int ]
  -> t
  -> batch_size:int
  -> batch_idx:int
  -> Tensor.t * Tensor.t

val batch_accuracy
  :  ?device:Torch_core.Device.t
  -> ?samples:int
  -> t
  -> [ `test | `train ]
  -> batch_size:int
  -> predict:(Tensor.t -> Tensor.t)
  -> float

val read_with_cache
  :  cache_file:string
  -> read:(unit -> t)
  -> t

val random_flip : Tensor.t -> Tensor.t
val random_crop : Tensor.t -> pad:int -> Tensor.t
