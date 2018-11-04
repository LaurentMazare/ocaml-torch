open Torch
val resnet18  : Var_store.t -> num_classes:int -> Layer.t_with_training
val resnet34  : Var_store.t -> num_classes:int -> Layer.t_with_training
val resnet50  : Var_store.t -> num_classes:int -> Layer.t_with_training
val resnet101 : Var_store.t -> num_classes:int -> Layer.t_with_training
val resnet152 : Var_store.t -> num_classes:int -> Layer.t_with_training
