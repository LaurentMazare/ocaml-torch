type t

(** [create name] creates a new language structure. *)
val create : name:string -> t

(** [add_word t word] adds a word (if not already present) to [t].
    The empty word is discarded.
*)
val add_word : t -> string -> unit

(** [add_sentence t str] adds all the words from [str] to [t] if needed.
*)
val add_sentence : t -> string -> unit

(** The start of string token. *)
val sos_token : t -> int

(** The end of string token. *)
val eos_token : t -> int

(** [length t] returns the number of currently stored tokens. *)
val length : t -> int

(** [name t] returns the language name as submitted to [create]. *)
val name : t -> string

(** [get_index t word] returns the index of [word] in [t] or [None] if
    [word] has not been added.
*)
val get_index : t -> string -> int option

(** [get_word t index] returns the word corresponding to [index] in [t].
*)
val get_word : t -> int -> string
