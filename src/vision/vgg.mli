open Torch

val vgg11    : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg11_bn : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg13    : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg13_bn : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg16    : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg16_bn : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg19    : Var_store.t -> num_classes:int -> Layer.t_with_training
val vgg19_bn : Var_store.t -> num_classes:int -> Layer.t_with_training
