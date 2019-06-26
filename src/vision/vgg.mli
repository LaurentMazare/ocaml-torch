open Base
open Torch

val vgg11 : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg11_bn : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg13 : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg13_bn : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg16 : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg16_bn : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg19 : Var_store.t -> num_classes:int -> Layer.t_with_training_f32
val vgg19_bn : Var_store.t -> num_classes:int -> Layer.t_with_training_f32

val vgg16_layers
  :  ?max_layer:int
  -> Var_store.t
  -> batch_norm:bool
  -> (Tensor.t_f32 -> (int, Tensor.t_f32, Int.comparator_witness) Map.t) Staged.t
