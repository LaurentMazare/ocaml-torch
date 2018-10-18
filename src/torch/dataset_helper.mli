type t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

val train_batch
  :  ?device:Torch_core.Device.t
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

val one_hot : (int, _, Bigarray.c_layout) Bigarray.Array1.t -> label_count:int -> Tensor.t

val read_with_cache
  :  cache_file:string
  -> read:(unit -> t)
  -> t
