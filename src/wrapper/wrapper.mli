module Kind : sig
  type t =
    | Uint8
    | Int8
    | Int16
    | Int
    | Int64
    | Half
    | Float
    | Double
    | ComplexHalf
    | ComplexFloat
    | ComplexDouble
end

module Tensor : sig
  type t

  (* TODO: maybe use a GADT for these subtypes of Kind ? *)
  val float_vec : ?kind:[ `half | `float | `double ] -> float list -> t
  val int_vec : ?kind:[ `uint8 | `int8 | `int16 | `int | `int64 ] -> int list -> t

  val zeros : ?kind:Kind.t -> int list -> t
  val ones : ?kind:Kind.t -> int list -> t
  val rand : int list -> t

  val shape : t -> int list
  val kind : t -> Kind.t
  val reshape : t -> int list -> t

  val sum : t -> t
  val mean : t -> t
  val neg : t -> t
  val log : t -> t
  val argmax : t -> t
  val softmax : t -> t

  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val pow : t -> t -> t
  val matmul : t -> t -> t
  val eq : t -> t -> t

  val sub_assign : t -> t -> unit

  val backward : t -> unit
  val grad : t -> t

  (* [set_requires_grad t ~b] modifies t and returns it back. *)
  val set_requires_grad : t -> b:bool -> t

  val get : t -> int -> t
  val select : t -> dim:int -> index:int -> t
  val float_value : t -> float
  val int_value : t -> int

  val set_float2 : t -> int -> int -> float -> unit

  val fill_float : t -> float -> unit
  val fill_int : t -> int -> unit

  val print : t -> unit
  val save : t -> string -> unit
  val load : string -> t
end
