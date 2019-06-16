type t

(** [create ~input_lang ~output_lang ~max_length] creates a new dataset
    from language [input_lang] to [output_lang]. Only texts with less
    than [max_length] words are included.
*)
val create : input_lang:string -> output_lang:string -> max_length:int -> t

(** [input_lang t] returns the input language. *)
val input_lang : t -> Lang.t

(** [output_lang t] returns the output language. *)
val output_lang : t -> Lang.t

(** [pairs t] returns the pairs of matched texts. The first element of the
    pair uses the input language, the right element the output language.
    The integers can be used with [Lang.t] to convert from/to words.
*)
val pairs : t -> (int list * int list) array

(** [reverse t] flips the input and output languages of a dataset. *)
val reverse : t -> t
