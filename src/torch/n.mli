type t
val root : t
val of_list : string list -> t
val to_string : t -> string
val of_string_parts : string -> t
val (/) : t -> string -> t
