open Torch

val squeezenet1_0 : Var_store.t -> num_classes:int -> ('a, 'a) Layer.t_with_training
val squeezenet1_1 : Var_store.t -> num_classes:int -> ('a, 'a) Layer.t_with_training
