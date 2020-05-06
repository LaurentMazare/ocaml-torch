include Base
include Torch_core.Wrapper.Tensor

let set_float2 t i j value = float_set t [ i; j ] value
let set_float1 t i value = float_set t [ i ] value
let set_int2 t i j value = int_set t [ i; j ] value
let set_int1 t i value = int_set t [ i ] value
let get_float2 t i j = float_get t [ i; j ]
let get_float1 t i = float_get t [ i ]
let get_int2 t i j = int_get t [ i; j ]
let get_int1 t i = int_get t [ i ]
let ( .%{} ) = int_get
let ( .%{}<- ) = int_set
let ( .%.{} ) = float_get
let ( .%.{}<- ) = float_set
let ( .%[] ) = get_int1
let ( .%[]<- ) = set_int1
let ( .%.[] ) = get_float1
let ( .%.[]<- ) = set_float1

let no_grad_ t ~f =
  if requires_grad t
  then (
    let t = set_requires_grad t ~r:false in
    Exn.protect
      ~f:(fun () -> f t)
      ~finally:(fun () -> ignore (set_requires_grad t ~r:true : t)))
  else f t

let no_grad f =
  let prev = grad_set_enabled false in
  Exn.protect ~f ~finally:(fun () -> ignore (grad_set_enabled prev : bool))

let zero_grad t =
  let grad = grad t in
  if defined grad
  then (
    ignore (detach_ grad : t);
    ignore (zero_ grad : t))

type create =
  ?requires_grad:bool
  -> ?kind:Torch_core.Kind.packed
  -> ?device:Device.t
  -> ?scale:float
  -> int list
  -> t

let type_ = kind
let to_type t ~type_ = totype t ~scalar_type:type_
let to_kind t ~kind = totype t ~scalar_type:kind

let to_device ?device t =
  match device with
  | None -> t
  | Some device -> to_ t ~device

let float_vec ?kind ?device dims = float_vec ?kind dims |> to_device ?device

let gen
    ~f
    ?(requires_grad = false)
    ?(kind = Torch_core.Kind.(T Float))
    ?(device = Device.Cpu)
    ?scale
    size
  =
  let t = f ~size ~options:(kind, device) in
  let t =
    Option.value_map
      scale
      ~f:(fun scale -> mul t (float_vec [ scale ] ~device))
      ~default:t
  in
  if requires_grad then set_requires_grad t ~r:true else t

let zeros = gen ~f:zeros
let ones = gen ~f:ones
let rand = gen ~f:rand
let randn = gen ~f:randn
let f v = float_vec [ v ] |> reshape ~shape:[]
let mm = matmul
let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( / ) = div
let ( ~- ) = neg
let ( -= ) t other = ignore (sub_ t other : t)
let ( += ) t other = ignore (add_ t other : t)
let ( /= ) t other = ignore (div_ t other : t)
let ( *= ) t other = ignore (mul_ t other : t)
let ( = ) = eq1
let pair_to_list (p1, p2) = [ p1; p2 ]

let conv2d ?(padding = 0, 0) ?(dilation = 1, 1) ?(groups = 1) input weight bias ~stride =
  conv2d
    input
    ~weight
    ~bias
    ~stride:(pair_to_list stride)
    ~padding:(pair_to_list padding)
    ~dilation:(pair_to_list dilation)
    ~groups

let conv_transpose2d
    ?(output_padding = 0, 0)
    ?(padding = 0, 0)
    ?(dilation = 1, 1)
    ?(groups = 1)
    input
    weight
    bias
    ~stride
  =
  conv_transpose2d
    input
    ~weight
    ~bias
    ~stride:(pair_to_list stride)
    ~padding:(pair_to_list padding)
    ~output_padding:(pair_to_list output_padding)
    ~groups
    ~dilation:(pair_to_list dilation)

let max_pool2d
    ?(padding = 0, 0)
    ?(dilation = 1, 1)
    ?(ceil_mode = false)
    ?stride
    self
    ~ksize
  =
  max_pool2d
    self
    ~kernel_size:(pair_to_list ksize)
    ~stride:(Option.value stride ~default:ksize |> pair_to_list)
    ~padding:(pair_to_list padding)
    ~dilation:(pair_to_list dilation)
    ~ceil_mode

let avg_pool2d
    ?(padding = 0, 0)
    ?(count_include_pad = false)
    ?(ceil_mode = false)
    ?stride
    ?divisor_override
    self
    ~ksize
  =
  let k1, k2 = ksize in
  let divisor_override = Option.value divisor_override ~default:Int.(k1 * k2) in
  avg_pool2d
    self
    ~kernel_size:(pair_to_list ksize)
    ~stride:(Option.value stride ~default:ksize |> pair_to_list)
    ~padding:(pair_to_list padding)
    ~ceil_mode
    ~count_include_pad
    ~divisor_override

let const_batch_norm ?(momentum = 0.1) ?(eps = 1e-5) input =
  batch_norm
    input
    ~weight:None
    ~bias:None
    ~running_mean:None
    ~running_var:None
    ~training:true
    ~momentum
    ~eps
    ~cudnn_enabled:false

let to_bigarray t ~kind =
  let bigarray = Bigarray.Genarray.create kind C_layout (shape t |> Array.of_list) in
  copy_to_bigarray (to_device t ~device:Cpu) bigarray;
  bigarray

let nll_loss ?(reduction = Torch_core.Reduction.Elementwise_mean) xs ~targets =
  nll_loss xs ~target:targets ~weight:None ~reduction ~ignore_index:(-100)

let cross_entropy_for_logits ?reduction logits ~targets =
  nll_loss ?reduction (log_softmax logits ~dim:(-1) ~dtype:(T Float)) ~targets

let dropout t ~p ~is_training = dropout t ~p ~train:is_training

let bce_loss ?(reduction = Torch_core.Reduction.Elementwise_mean) t ~targets =
  binary_cross_entropy t ~target:targets ~weight:None ~reduction

let mse_loss ?(reduction = Torch_core.Reduction.Elementwise_mean) t1 t2 =
  mse_loss t1 ~target:t2 ~reduction

let huber_loss ?(reduction = Torch_core.Reduction.Elementwise_mean) t1 t2 =
  let d = abs (t1 - t2) in
  let half = f 0.5 in
  let err = where1 ~condition:(le d (Scalar.float 1.)) (half * d * d) (d - half) in
  match reduction with
  | None -> err
  | Elementwise_mean -> mean err
  | Sum -> sum err

let bce_loss_with_logits ?(reduction = Torch_core.Reduction.Elementwise_mean) t ~targets =
  let max_val = clamp_min_ (-t) ~min:(Scalar.float 0.) in
  let one_minus_targets = ones_like targets - targets in
  let bce =
    add_
      (add_ (mul_ one_minus_targets t) max_val)
      (add_ (exp_ (-max_val)) (exp_ (-t - max_val)) |> log_)
  in
  match reduction with
  | None -> bce
  | Elementwise_mean -> mean bce
  | Sum -> sum bce

let pp formatter t =
  let shape = shape t in
  let element_count = List.fold shape ~init:1 ~f:Int.( * ) in
  if element_count < 1_000
  then (
    Caml.Format.pp_print_newline formatter ();
    Caml.Format.pp_print_string formatter (to_string t ~line_size:96);
    Caml.Format.pp_print_newline formatter ())
  else
    List.map shape ~f:Int.to_string
    |> String.concat ~sep:", "
    |> Printf.sprintf "Tensor<%s>"
    |> Caml.Format.pp_print_string formatter

let copy t =
  let t_ = zeros (shape t) ~kind:(kind t) in
  copy_ t_ ~src:t;
  t_

let shape_str t = List.map (shape t) ~f:Int.to_string |> String.concat ~sep:", "
let print_shape ?(name = "") t = Stdio.printf "%s<%s>\n%!" name (shape_str t)

let bigarray_to_array1 bigarray ~f =
  try
    let bigarray = Bigarray.array1_of_genarray bigarray in
    Array.init (Bigarray.Array1.dim bigarray) ~f:(fun i -> f bigarray.{i}) |> Option.some
  with
  | Invalid_argument _ -> None

let bigarray_to_array2 bigarray ~f =
  try
    let bigarray = Bigarray.array2_of_genarray bigarray in
    Array.init (Bigarray.Array2.dim1 bigarray) ~f:(fun i ->
        Array.init (Bigarray.Array2.dim2 bigarray) ~f:(fun j -> f bigarray.{i, j}))
    |> Option.some
  with
  | Invalid_argument _ -> None

let bigarray_to_array3 bigarray ~f =
  try
    let bigarray = Bigarray.array3_of_genarray bigarray in
    Array.init (Bigarray.Array3.dim1 bigarray) ~f:(fun i ->
        Array.init (Bigarray.Array3.dim2 bigarray) ~f:(fun j ->
            Array.init (Bigarray.Array3.dim3 bigarray) ~f:(fun k -> f bigarray.{i, j, k})))
    |> Option.some
  with
  | Invalid_argument _ -> None

let to_float1 t =
  match kind t with
  | T Float -> to_bigarray t ~kind:Bigarray.float32 |> bigarray_to_array1 ~f:Fn.id
  | T Double -> to_bigarray t ~kind:Bigarray.float64 |> bigarray_to_array1 ~f:Fn.id
  | _ -> None

let to_float2 t =
  match kind t with
  | T Float -> to_bigarray t ~kind:Bigarray.float32 |> bigarray_to_array2 ~f:Fn.id
  | T Double -> to_bigarray t ~kind:Bigarray.float64 |> bigarray_to_array2 ~f:Fn.id
  | _ -> None

let to_float3 t =
  match kind t with
  | T Float -> to_bigarray t ~kind:Bigarray.float32 |> bigarray_to_array3 ~f:Fn.id
  | T Double -> to_bigarray t ~kind:Bigarray.float64 |> bigarray_to_array3 ~f:Fn.id
  | _ -> None

let to_int1 t =
  match kind t with
  | T Int -> to_bigarray t ~kind:Bigarray.int32 |> bigarray_to_array1 ~f:Int32.to_int_exn
  | T Int64 ->
    to_bigarray t ~kind:Bigarray.int64 |> bigarray_to_array1 ~f:Int64.to_int_exn
  | _ -> None

let to_int2 t =
  match kind t with
  | T Int -> to_bigarray t ~kind:Bigarray.int32 |> bigarray_to_array2 ~f:Int32.to_int_exn
  | T Int64 ->
    to_bigarray t ~kind:Bigarray.int64 |> bigarray_to_array2 ~f:Int64.to_int_exn
  | _ -> None

let to_int3 t =
  match kind t with
  | T Int -> to_bigarray t ~kind:Bigarray.int32 |> bigarray_to_array3 ~f:Int32.to_int_exn
  | T Int64 ->
    to_bigarray t ~kind:Bigarray.int64 |> bigarray_to_array3 ~f:Int64.to_int_exn
  | _ -> None

let to_int1_exn t = Option.value_exn (to_int1 t)
let to_int2_exn t = Option.value_exn (to_int2 t)
let to_int3_exn t = Option.value_exn (to_int3 t)
let to_float1_exn t = Option.value_exn (to_float1 t)
let to_float2_exn t = Option.value_exn (to_float2 t)
let to_float3_exn t = Option.value_exn (to_float3 t)
let to_float0_exn = float_value

let to_float0 t =
  try float_value t |> Option.some with
  | _ -> None

let to_int0_exn = int_value

let to_int0 t =
  try int_value t |> Option.some with
  | _ -> None

let of_bigarray ?device ba = of_bigarray ba |> to_device ?device

let of_float0 ?device f =
  Bigarray.Array0.of_value Float32 C_layout f
  |> Bigarray.genarray_of_array0
  |> of_bigarray ?device

let of_float1 ?device f =
  Bigarray.Array1.of_array Float32 C_layout f
  |> Bigarray.genarray_of_array1
  |> of_bigarray ?device

let of_float2 ?device f =
  Bigarray.Array2.of_array Float32 C_layout f
  |> Bigarray.genarray_of_array2
  |> of_bigarray ?device

let of_float3 ?device f =
  Bigarray.Array3.of_array Float32 C_layout f
  |> Bigarray.genarray_of_array3
  |> of_bigarray ?device

let of_int0 ?device f =
  Bigarray.Array0.of_value Int C_layout f
  |> Bigarray.genarray_of_array0
  |> of_bigarray ?device

let of_int1 ?device f =
  Bigarray.Array1.of_array Int C_layout f
  |> Bigarray.genarray_of_array1
  |> of_bigarray ?device

let of_int2 ?device f =
  Bigarray.Array2.of_array Int C_layout f
  |> Bigarray.genarray_of_array2
  |> of_bigarray ?device

let of_int3 ?device f =
  Bigarray.Array3.of_array Int C_layout f
  |> Bigarray.genarray_of_array3
  |> of_bigarray ?device

let minimum t = reshape t ~shape:[ -1 ] |> min_values ~dim:[ 0 ] ~keepdim:false
let maximum t = reshape t ~shape:[ -1 ] |> max_values ~dim:[ 0 ] ~keepdim:false

let flatten t =
  let batch_size = shape t |> List.hd_exn in
  view t ~size:[ batch_size; -1 ]

let squeeze_last t = squeeze1 t ~dim:(-1)
let scale t f = mul1 t (Scalar.float f)
let eq_scalar = eq

let eq t1 t2 =
  if Torch_core.Kind.( <> ) (kind t1) (kind t2)
  then false
  else if Caml.( <> ) (shape t1) (shape t2)
  then false
  else eq1 t1 t2 |> all |> to_int0_exn |> ( <> ) 0

let to_list t =
  let size =
    match size t with
    | [] -> failwith "scalar tensor"
    | size :: _ -> size
  in
  List.init size ~f:(get t)
