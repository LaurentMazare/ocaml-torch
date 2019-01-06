open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  let manual_seed = foreign "at_manual_seed" (int64_t @-> returning void)
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let new_tensor = foreign "at_new_tensor" (void @-> returning t)

    let tensor_of_data =
      foreign "at_tensor_of_data"
        (   ptr void (* data *)
        @-> ptr long (* dims *)
        @-> int      (* ndims *)
        @-> int      (* element size in bytes *)
        @-> int      (* kind *)
        @-> returning t)

    let copy_data =
      foreign "at_copy_data"
        (   t        (* tensor *)
        @-> ptr void (* data *)
        @-> int64_t  (* numel *)
        @-> int      (* element size in bytes *)
        @-> returning void)

    let copy_ =
      foreign "at_copy_"
        (   t (* dst *)
        @-> t (* src *)
        @-> returning void)

    let float_vec =
      foreign "at_float_vec"
        (   ptr double (* values *)
        @-> int        (* num values *)
        @-> int        (* kind *)
        @-> returning t)

    let int_vec =
      foreign "at_int_vec"
        (   ptr int64_t (* values *)
        @-> int         (* num values *)
        @-> int         (* kind *)
        @-> returning t)

    let defined = foreign "at_defined" (t @-> returning bool)
    let num_dims = foreign "at_dim" (t @-> returning int)

    let shape =
      foreign "at_shape"
        (   t
        @-> ptr int  (* dims *)
        @-> returning void)

    let scalar_type = foreign "at_scalar_type" (t @-> returning int)

    let backward = foreign "at_backward" (t @-> int @-> int @-> returning void)
    let requires_grad =
      foreign "at_requires_grad" (t @-> returning int)
    let grad_set_enabled =
      foreign "at_grad_set_enabled" (int @-> returning int)

    let get = foreign "at_get" (t @-> int @-> returning t)
    let double_value =
      foreign
        "at_double_value_at_indexes"
        (t @-> ptr int @-> int @-> returning float)

    let int64_value =
      foreign
        "at_int64_value_at_indexes"
        (t @-> ptr int @-> int @-> returning int64_t)

    let double_value_set =
      foreign
        "at_set_double_value_at_indexes"
        (t @-> ptr int @-> int @-> float @-> returning void)

    let int64_value_set =
      foreign
        "at_set_int64_value_at_indexes"
        (t @-> ptr int @-> int @-> int64_t @-> returning void)

    let fill_double = foreign "at_fill_double" (t @-> float @-> returning void)
    let fill_int64 = foreign "at_fill_int64" (t @-> int64_t @-> returning void)

    let print = foreign "at_print" (t @-> returning void)
    let to_string = foreign "at_to_string" (t @-> int @-> returning string)
    let free = foreign "at_free" (t @-> returning void)

    let run_backward =
      foreign "at_run_backward"
        (   ptr t
        @-> int
        @-> ptr t
        @-> int
        @-> ptr t
        @-> int
        @-> int
        @-> returning void)
  end

  module Scalar = struct
    type t = unit ptr
    let t : t typ = ptr void
    let int = foreign "ats_int" (int64_t @-> returning t)
    let float = foreign "ats_float" (float @-> returning t)
    let free = foreign "ats_free" (t @-> returning void)
  end

  module Serialize = struct
    let t = Tensor.t
    let save = foreign "at_save" (t @-> string @-> returning void)
    let load = foreign "at_load" (string @-> returning t)
    let save_multi =
      foreign "at_save_multi" (ptr t @-> ptr (ptr char) @-> int @-> string @-> returning void)
    let load_multi =
      foreign "at_load_multi" (ptr t @-> ptr (ptr char) @-> int @-> string @-> returning void)
    let load_multi_ =
      foreign "at_load_multi_" (ptr t @-> ptr (ptr char) @-> int @-> string @-> returning void)
    let load_callback =
      foreign
        "at_load_callback"
        (string
        @-> static_funptr Ctypes.(string @-> t @-> returning void)
        @-> returning void)
  end

  module Optimizer = struct
    type t = unit ptr
    let t : t typ = ptr void

    let adam =
      foreign "ato_adam" (float @-> float @-> float @-> float @-> returning t)
    let rmsprop =
      foreign "ato_rmsprop"
        (   float (* learning rate *)
        @-> float (* alpha *)
        @-> float (* eps *)
        @-> float (* weight decay *)
        @-> float (* momentum *)
        @-> int   (* centered *)
        @-> returning t)

    let sgd =
      foreign "ato_sgd"
        (   float (* learning rate *)
        @-> float (* momentum *)
        @-> float (* dampening *)
        @-> float (* weight decay *)
        @-> bool  (* nesterov *)
        @-> returning t)
    let add_parameters =
      foreign "ato_add_parameters" (t @-> ptr Tensor.t @-> int @-> returning void)
    let set_learning_rate = foreign "ato_set_learning_rate" (t @-> float @-> returning void)
    let set_momentum = foreign "ato_set_momentum" (t @-> float @-> returning void)
    let zero_grad = foreign "ato_zero_grad" (t @-> returning void)
    let step = foreign "ato_step" (t @-> returning void)
    let free = foreign "ato_free" (t @-> returning void)
  end

  module Cuda = struct
    let device_count = foreign "atc_cuda_device_count" (void @-> returning int)
    let is_available = foreign "atc_cuda_is_available" (void @-> returning int)
    let cudnn_is_available = foreign "atc_cudnn_is_available" (void @-> returning int)
    let set_benchmark_cudnn = foreign "atc_set_benchmark_cudnn" (int @-> returning void)
  end

  module Ivalue = struct
    type t = unit ptr
    let t : t typ = ptr void

    let to_int64 = foreign "ati_to_int" (t @-> returning int64_t)
    let to_double = foreign "ati_to_double" (t @-> returning double)
    let to_tensor = foreign "ati_to_tensor" (t @-> returning Tensor.t)
    let tuple_length = foreign "ati_tuple_length" (t @-> returning int)
    let to_tuple = foreign "ati_to_tuple" (t @-> ptr t @-> int @-> returning void)

    let tensor = foreign "ati_tensor" (Tensor.t @-> returning t)
    let int64 = foreign "ati_int" (int64_t @-> returning t)
    let double = foreign "ati_double" (float @-> returning t)
    let tuple = foreign "ati_tuple" (ptr t @-> int @-> returning t)

    let tag = foreign "ati_tag" (t @-> returning int)

    let free = foreign "ati_free" (t @-> returning void)
  end

  module Module = struct
    type t = unit ptr
    let t : t typ = ptr void
    let load = foreign "atm_load" (string @-> returning t)
    let forward = foreign "atm_forward" (t @-> ptr Tensor.t @-> int @-> returning Tensor.t)
    let forward_ =
      foreign "atm_forward_"
        (   t
        @-> ptr Ivalue.t
        @-> int
        @-> returning Ivalue.t)
    let free = foreign "atm_free" (t @-> returning void)
  end

  module TensorG = Torch_bindings_generated.C(F)
end
