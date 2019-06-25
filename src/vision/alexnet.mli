open Torch

(* If [num_classes] is provided, the last layer is a fully-connected layer with
   [num_classes] output. When [num_classes] is not provided the values before
   this last layer are returned - this is useful to use these models as feature
   extractors.
*)
val alexnet : ?num_classes:int -> Var_store.t -> ('a, 'a) Layer.t_with_training
