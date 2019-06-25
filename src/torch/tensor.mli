open Torch_core

(* TODO: proper types for Tensor1D, Tensor2D, Tensor3D, ... ? *)
(* TODO: GADT for array element types ? *)
type 'a t = 'a Torch_core.Wrapper.Tensor.t

include module type of Torch_core.Wrapper.Tensor with type 'a t := 'a t

(** [set_float2 t i j v] sets the element at index [i] and [j] of
    bidimensional tensor [t] to [v].
*)
val set_float2 : _ t -> int -> int -> float -> unit

(** [set_float1 t i v] sets the element at index [i] of single
    dimension tensor [t] to [v].
*)
val set_float1 : _ t -> int -> float -> unit

(** [set_int2 t i j v] sets the element at index [i] and [j] of
    bidimensional tensor [t] to [v].
*)
val set_int2 : _ t -> int -> int -> int -> unit

(** [set_int1 t i v] sets the element at index [i] of single
    dimension tensor [t] to [v].
*)
val set_int1 : _ t -> int -> int -> unit

(** [get_float2 t i j] returns the current value from bidimensional tensor [t]
    at index [i] and [j].
*)
val get_float2 : _ t -> int -> int -> float

(** [get_float1 t i j] returns the current value from single dimension tensor [t]
    at index [i].
*)
val get_float1 : _ t -> int -> float

(** [get_int2 t i j] returns the current value from bidimensional tensor [t]
    at indexex [i] and [j].
*)
val get_int2 : _ t -> int -> int -> int

(** [get_int1 t i j] returns the current value from single dimension tensor [t]
    at index [i].
*)
val get_int1 : _ t -> int -> int

(** Gets an integer element from an arbitrary dimension tensor. *)
val ( .%{} ) : _ t -> int list -> int

(** Sets an integer element on an arbitrary dimension tensor. *)
val ( .%{}<- ) : _ t -> int list -> int -> unit

(** Gets a float element from an arbitrary dimension tensor. *)
val ( .%.{} ) : _ t -> int list -> float

(** Sets a float element on an arbitrary dimension tensor. *)
val ( .%.{}<- ) : _ t -> int list -> float -> unit

(** Gets an integer element from a single dimension tensor. *)
val ( .%[] ) : _ t -> int -> int

(** Sets an integer element on a single dimension tensor. *)
val ( .%[]<- ) : _ t -> int -> int -> unit

(** Gets a float element from a single dimension tensor. *)
val ( .%.[] ) : _ t -> int -> float

(** Sets a float element on a single dimension tensor. *)
val ( .%.[]<- ) : _ t -> int -> float -> unit

(** [no_grad_ t ~f] runs [f] on [t] without tracking gradients for t. *)
val no_grad_ : 'b t -> f:('b t -> 'a) -> 'a

val no_grad : (unit -> 'a) -> 'a
val zero_grad : _ t -> unit

(** Pointwise addition. *)
val ( + ) : 'a t -> 'a t -> 'a t

(** Pointwise substraction. *)
val ( - ) : 'a t -> 'a t -> 'a t

(** Pointwise multiplication. *)
val ( * ) : 'a t -> 'a t -> 'a t

(** Pointwise division. *)
val ( / ) : 'a t -> 'a t -> 'a t

(** [t += u] modifies [t] by adding values from [u] in a pointwise way. *)
val ( += ) : 'a t -> 'a t -> unit

(** [t -= u] modifies [t] by subtracting values from [u] in a pointwise way. *)
val ( -= ) : 'a t -> 'a t -> unit

(** [t *= u] modifies [t] by multiplying values from [u] in a pointwise way. *)
val ( *= ) : 'a t -> 'a t -> unit

(** [t /= u] modifies [t] by dividing values from [u] in a pointwise way. *)
val ( /= ) : 'a t -> 'a t -> unit

(** [~-u] returns the opposite of [t], i.e. the same as [Tensor.(f 0. - t)]. *)
val ( ~- ) : 'a t -> 'a t

(** Pointwise equality. *)
val ( = ) : 'a t -> 'a t -> 'a t

(** [eq t1 t2] returns true if [t1] and [t2] have the same kind, shape, and
    all their elements are identical.
*)
val eq : 'a t -> 'a t -> bool

(** [mm t1 t2] returns the dot product or matrix multiplication between [t1] and [t2]. *)
val mm : 'a t -> 'a t -> 'a t

(** [f v] returns a scalar tensor with value [v]. *)
val f : float -> _ t

type 'a create =
  ?requires_grad:bool
  -> ?kind:Torch_core.Kind.packed
  -> ?device:Device.t
  -> ?scale:float
  -> int list
  -> 'a t

(** Creates a tensor with value 0. *)
val zeros : _ create

(** Creates a tensor with value 1. *)
val ones : _ create

(** Creates a tensor with random values sampled uniformly between 0 and 1. *)
val rand : _ create

(** Creates a tensor with random values sampled using a standard normal distribution. *)
val randn : _ create

(** Creates a tensor from a list of float values. *)
val float_vec : ?kind:[ `double | `float | `half ] -> ?device:Device.t -> float list -> _ t

(** [to_type t ~type_] returns a tensor similar to [t] but converted to kind [type_]. *)
val to_type : _ t -> type_:Kind.packed -> _ t

(** [to_kind t ~kind] returns a tensor similar to [t] but converted to kind [kind]. *)
val to_kind : _ t -> kind:Kind.packed -> _ t

(** [kind t] returns the kind of elements hold in tensor [t]. *)
val type_ : _ t -> Kind.packed

(** [to_device t ~device] returns a tensor identical to [t] but placed on device [device]. *)
val to_device : ?device:Device.t -> 'a t -> 'a t

(** [to_float0 t] returns the value hold in a scalar (0-dimension) tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_float0 : _ t -> float option

(** [to_float1 t] returns the array of values hold in a single dimension tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_float1 : _ t -> float array option

(** [to_float2 t] returns the array of values hold in a bidimensional tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_float2 : _ t -> float array array option

(** [to_float3 t] returns the array of values hold in a tridimensional tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_float3 : _ t -> float array array array option

(** [to_float0_exn t] returns the value hold in a scalar (0-dimension) tensor. *)
val to_float0_exn : _ t -> float

(** [to_float1_exn t] returns the array of values hold in a single dimension tensor. *)
val to_float1_exn : _ t -> float array

(** [to_float2_exn t] returns the array of values hold in a bidimensional tensor. *)
val to_float2_exn : _ t -> float array array

(** [to_float3_exn t] returns the array of values hold in a tridimensional tensor. *)
val to_float3_exn : _ t -> float array array array

(** [to_int0 t] returns the value hold in a scalar (0-dimension) tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_int0 : _ t -> int option

(** [to_int1 t] returns the array of values hold in a single dimension tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_int1 : _ t -> int array option

(** [to_int2 t] returns the array of values hold in a bidimensional tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_int2 : _ t -> int array array option

(** [to_int3 t] returns the array of values hold in a tridimensional tensor.
    If the dimension are incorrect, [None] is returned.
*)
val to_int3 : _ t -> int array array array option

(** [to_int0_exn t] returns the value hold in a scalar (0-dimension) tensor. *)
val to_int0_exn : _ t -> int

(** [to_int1_exn t] returns the array of values hold in a single dimension tensor. *)
val to_int1_exn : _ t -> int array

(** [to_int2_exn t] returns the array of values hold in a bidimensional tensor. *)
val to_int2_exn : _ t -> int array array

(** [to_int3_exn t] returns the array of values hold in a tridimensional tensor. *)
val to_int3_exn : _ t -> int array array array

(** [of_float0 v] creates a scalar (0-dimension) tensor with value v. *)
val of_float0 : ?device:Device.t -> float -> _ t

(** [of_float1 v] creates a single dimension tensor with values vs. *)
val of_float1 : ?device:Device.t -> float array -> _ t

(** [of_float2 v] creates a two dimension tensor with values vs. *)
val of_float2 : ?device:Device.t -> float array array -> _ t

(** [of_float3 v] creates a three dimension tensor with values vs. *)
val of_float3 : ?device:Device.t -> float array array array -> _ t

(** [of_int0 v] creates a scalar (0-dimension) tensor with value v. *)
val of_int0 : ?device:Device.t -> int -> _ t

(** [of_int1 v] creates a single dimension tensor with values vs. *)
val of_int1 : ?device:Device.t -> int array -> _ t

(** [of_int2 v] creates a two dimension tensor with values vs. *)
val of_int2 : ?device:Device.t -> int array array -> _ t

(** [of_int3 v] creates a three dimension tensor with values vs. *)
val of_int3 : ?device:Device.t -> int array array array -> _ t

val conv2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?groups:int
  -> 'a t (* input *)
  -> 'a t (* weight *)
  -> 'a t option (* bias *)
  -> stride:int * int
  -> 'a t

val conv_transpose2d
  :  ?output_padding:int * int
  -> ?padding:int * int
  -> ?dilation:int * int
  -> ?groups:int
  -> 'a t (* input *)
  -> 'a t (* weight *)
  -> 'a t option (* bias *)
  -> stride:int * int
  -> 'a t

val max_pool2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> 'a t
  -> ksize:int * int
  -> 'a t

val avg_pool2d
  :  ?padding:int * int
  -> ?count_include_pad:bool
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> 'a t
  -> ksize:int * int
  -> 'a t

val const_batch_norm : ?momentum:float -> ?eps:float -> 'a t -> 'a t

(** [of_bigarray ba] returns a tensor which shape and kind are based on
    [ba] and holding the same data.
*)
val of_bigarray : ?device:Device.t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> _ t

(** [copy_to_bigarray t ba] copies the data from [t] to [ba]. The dimensions of
    [ba] and its kind of element must match the dimension and kind of [t].
*)
val copy_to_bigarray : _ t -> ('b, 'a, Bigarray.c_layout) Bigarray.Genarray.t -> unit

(** [to_bigarray t ~kind] converts [t] to a bigarray using the c layout. [kind] has
    to be compatible with the element kind of [t].
*)
val to_bigarray
  :  _ t
  -> kind:('a, 'b) Bigarray.kind
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

val cross_entropy_for_logits : ?reduction:Reduction.t -> _ t -> targets:_ t -> _ t

(** [dropout t ~p ~is_training] applies dropout to [t] with probability [p].
    If [is_training] is [false], [t] is returned.
    If [is_training] is [true], a tensor similar to [t] is returned except that
    each element has a probability [p] to be replaced by [0].
*)
val dropout : 'a t -> p:float (* dropout prob *) -> is_training:bool -> 'a t

val nll_loss : ?reduction:Torch_core.Reduction.t -> _ t -> targets:_ t -> _ t

(** [bce_loss t ~targets] returns the binary cross entropy loss between [t]
    and [targets]. Elements of [t] are supposed to represent a probability
    distribution (according to the last dimension of [t]), so should be
    between 0 and 1 and sum to 1. *)
val bce_loss : ?reduction:Torch_core.Reduction.t -> _ t -> targets:_ t -> _ t

(** [bce_loss_with_logits t ~targets] returns the binary cross entropy loss between [t]
    and [targets]. Elements of [t] are logits, a softmax is used in this function to
    convert them to a probability distribution. *)
val bce_loss_with_logits : ?reduction:Torch_core.Reduction.t -> _ t -> targets:_ t -> _ t

(** [mse_loss t1 t2] returns the square of the difference between [t1] and [t2].
    [reduction] can be used to either keep the whole tensor or reduce it by averaging
    or summing.
*)

val mse_loss : ?reduction:Torch_core.Reduction.t -> 'a t -> 'a t -> 'a t
val huber_loss : ?reduction:Torch_core.Reduction.t -> 'a t -> 'a t -> 'a t

(** [pp] is a pretty-printer for tensors to be used in top-levels such as utop
    or jupyter.
*)
val pp : Format.formatter -> _ t -> unit
  [@@ocaml.toplevel_printer]

(** [copy t] returns a new copy of [t] with the same size and data which does
    not share storage with t.
*)
val copy : 'a t -> 'a t

(** [shape_str t] returns the shape/size of the current tensor as a string.
    This is useful for pretty printing.
*)
val shape_str : _ t -> string

(** [print_shape ?name t] prints the shape/size of t on stdout. If [name] is
    provided, this is also  printed.
*)
val print_shape : ?name:string -> _ t -> unit

(** [minimum t] returns the minimum element of tensor [t]. *)
val minimum : 'a t -> 'a t

(** [maximum t] returns the maximum element of tensor [t]. *)
val maximum : 'a t -> 'a t

(** [flatten t] returns a flattened version of t, i.e. a single
    dimension version of the tensor.
    This is equivalent to [Tensor.view t ~size:[-1]].
*)
val flatten : 'a t -> 'a t

(** [squeeze_last t] squeezes the last dimension of t, i.e. if this
    dimension has a size of 1 it is removed.
*)
val squeeze_last : 'a t -> 'a t

(** [scale t f] returns the result of multiplying tensor t by f. *)
val scale : 'a t -> float -> 'a t

(** [to_list t] returns the list of tensors extracted from the first dimension.
    This is the inverse of [cat ~dim:0]. *)
val to_list : 'a t -> 'a t list
