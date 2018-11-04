include Base
include Torch_core.Wrapper.Tensor

(* TODO: implement these in a single C call rather than chaining [get]. *)
let set_float2 t i j value = fill_float (get (get t i) j) value
let set_float1 t i value = fill_float (get t i) value
let set_int2 t i j value = fill_int (get (get t i) j) value
let set_int1 t i value = fill_int (get t i) value

let get_float2 t i j = float_value (get (get t i) j)
let get_float1 t i = float_value (get t i)
let get_int2 t i j = int_value (get (get t i) j)
let get_int1 t i = int_value (get t i)

let no_grad t ~f =
  if requires_grad t
  then
    let t = set_requires_grad t ~r:false in
    let result = f t in
    ignore (set_requires_grad t ~r:true : t);
    result
  else f t

let zero_grad t =
  let grad = grad t in
  if defined grad
  then begin
    ignore (detach_ grad : t);
    ignore (zero_ grad : t)
  end

type create
  =  ?requires_grad:bool
  -> ?kind:Torch_core.Kind.t
  -> ?device:Torch_core.Device.t
  -> ?scale:float
  -> int list
  -> t

let to_type t ~type_ = totype t ~scalar_type:type_

let to_device ?device t =
  match device with
  | None -> t
  | Some device -> to1 t ~device

let float_vec ?kind ?device dims =
  float_vec ?kind dims |> to_device ?device

let gen ~f ?(requires_grad = false) ?(kind = Torch_core.Kind.Float) ?(device = Torch_core.Device.Cpu) ?scale size =
  let t = f ~size ~options:(kind, device) in
  let t =
    Option.value_map scale
      ~f:(fun scale -> mul t (float_vec [ scale ] ~device))
      ~default:t
  in
  if requires_grad
  then set_requires_grad t ~r:true
  else t

let zeros = gen ~f:zeros
let ones = gen ~f:ones
let rand = gen ~f:rand
let randn = gen ~f:randn

let f v = float_vec [ v ] |> reshape ~shape:[]
let mm = matmul

let (+) = add
let (-) = sub
let ( * ) = mul
let (/) = div

let (~-) = neg
let (-=) t other = ignore (sub_ t other : t)
let (+=) t other = ignore (add_ t other : t)
let (/=) t other = ignore (div_ t other : t)
let ( *=) t other = ignore (mul_ t other : t)
let (=) = eq

let pair_to_list (p1, p2) = [ p1; p2 ]
let conv2d ?(padding=0, 0) ?(dilation=1, 1) ?(groups=1) input weight bias ~stride =
  conv2d input
    ~weight
    ~bias
    ~stride:(pair_to_list stride)
    ~padding:(pair_to_list padding)
    ~dilation:(pair_to_list dilation)
    ~groups

let conv_transpose2d
    ?(output_padding=0, 0)
    ?(padding=0, 0)
    ?(dilation=1, 1)
    ?(groups=1)
    input
    weight
    bias
    ~stride
  =
  conv_transpose2d input
    ~weight
    ~bias
    ~stride:(pair_to_list stride)
    ~padding:(pair_to_list padding)
    ~output_padding:(pair_to_list output_padding)
    ~groups
    ~dilation:(pair_to_list dilation)

let max_pool2d ?(padding=0, 0) ?(dilation=1, 1) ?(ceil_mode=false) ?stride self ~ksize =
  max_pool2d self
    ~kernel_size:(pair_to_list ksize)
    ~stride:(Option.value stride ~default:ksize |> pair_to_list)
    ~padding:(pair_to_list padding)
    ~dilation:(pair_to_list dilation)
    ~ceil_mode

let avg_pool2d ?(padding=0, 0) ?(count_include_pad=false) ?(ceil_mode=false) ?stride self ~ksize =
  avg_pool2d self
    ~kernel_size:(pair_to_list ksize)
    ~stride:(Option.value stride ~default:ksize |> pair_to_list)
    ~padding:(pair_to_list padding)
    ~ceil_mode
    ~count_include_pad

let const_batch_norm ?(momentum=0.1) ?(eps=1e-5) input =
  batch_norm input
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
  copy_to_bigarray t bigarray;
  bigarray

let undefined = lazy (new_tensor ())

let nll_loss ?(reduction=Torch_core.Reduction.Elementwise_mean) xs ~targets =
  nll_loss xs
    ~target:targets
    ~weight:(Lazy.force undefined)
    ~reduction:(Torch_core.Reduction.to_int reduction)
    ~ignore_index:(-100)

let cross_entropy_for_logits ?reduction logits ~targets =
  nll_loss ?reduction (log_softmax logits ~dim:(-1)) ~targets

let dropout t ~p ~is_training = dropout t ~p ~train:is_training

let bce_loss ?(reduction=Torch_core.Reduction.Elementwise_mean) t ~targets =
  binary_cross_entropy t
    ~target:targets
    ~weight:(Lazy.force undefined)
    ~reduction:(Torch_core.Reduction.to_int reduction)

let pp formatter t =
  let shape = shape t in
  let element_count = List.fold shape ~init:1 ~f:Int.( * ) in
  if element_count < 1_000
  then begin
    Caml.Format.pp_print_newline formatter ();
    Caml.Format.pp_print_string formatter (to_string t ~line_size:96);
    Caml.Format.pp_print_newline formatter ()
  end else begin
    List.map shape ~f:Int.to_string
    |> String.concat ~sep:", "
    |> Printf.sprintf "Tensor<%s>"
    |> Caml.Format.pp_print_string formatter
  end

let copy t =
  let t_ = view t ~size:(shape t) in
  copy_ t_ ~src:t;
  t_
