open Base
open Torch

val vgg11 : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg11_bn : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg13 : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg13_bn : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg16 : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg16_bn : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg19 : Var_store.t -> num_classes:int -> Layer.f32_with_training
val vgg19_bn : Var_store.t -> num_classes:int -> Layer.f32_with_training

val vgg16_layers
  :  ?max_layer:int
  -> Var_store.t
  -> batch_norm:bool
  -> (Tensor.f32 -> (int, Tensor.f32, Int.comparator_witness) Map.t) Staged.t
