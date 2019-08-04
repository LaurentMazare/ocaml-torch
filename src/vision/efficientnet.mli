(* EfficientNet models, https://arxiv.org/abs/1905.11946
   The implementation is very similar to:
     https://github.com/lukemelas/EfficientNet-PyTorch
*)
open Torch

val b0 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b1 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b2 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b3 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b4 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b5 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b6 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val b7 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
