type t

val create : input_lang:string -> output_lang:string -> t
val input_lang : t -> Lang.t
val output_lang : t -> Lang.t
val pairs : t -> (string * string) list
val reverse : t -> t
