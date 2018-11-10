open Torch

type t =
  { model_name : string
  ; model : Layer.t_with_training
  ; epochs : int
  ; lr_schedule : batch_idx:int -> batches_per_epoch:int -> epoch_idx:int -> float
  ; batch_size : int
  }
