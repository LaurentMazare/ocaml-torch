open Torch
val resnet18  : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val resnet34  : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val resnet50  : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val resnet101 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
val resnet152 : ?num_classes:int -> Var_store.t -> Layer.t_with_training
