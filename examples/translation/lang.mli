type t

val create : name:string -> t
val add_word : t -> string -> unit
val add_sentence : t -> string -> unit
val sos_token : t -> int
val eos_token : t -> int
val length : t -> int
val name : t -> string
