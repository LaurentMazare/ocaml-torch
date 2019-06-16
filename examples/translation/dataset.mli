type t

val create : input_lang:string -> output_lang:string -> max_length:int -> t
val input_lang : t -> Lang.t
val output_lang : t -> Lang.t
val pairs : t -> (int list * int list) array
val reverse : t -> t
