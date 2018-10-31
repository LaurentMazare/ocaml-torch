type t
val of_list : string list -> t
val to_string : t -> string
val (/) : t -> string -> t

(** [default t str] builds a default name based on [str] when
    [t] is [None], otherwise [t] is used.
*)
val default : t option -> string -> t
