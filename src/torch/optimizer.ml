open Base
module Optimizer = Torch_core.Wrapper.Optimizer

module Clip_grad = struct
  type t =
    | Norm2 of float
    | Value of float
end

type t =
  { optimizer : Optimizer.t
  ; trainable_vars : Tensor.packed list
  }

let create optimizer ~vs =
  let trainable_vars = Var_store.trainable_vars vs in
  (* TODO: get back to doing a single call. *)
  List.iter trainable_vars ~f:(fun (Tensor.T v) ->
    Optimizer.add_parameters optimizer [ v ]);
  { optimizer; trainable_vars }

let adam ?(beta1 = 0.9) ?(beta2 = 0.999) ?(weight_decay = 0.) vs ~learning_rate =
  Optimizer.adam ~learning_rate ~beta1 ~beta2 ~weight_decay |> create ~vs

let rmsprop
    ?(alpha = 0.99)
    ?(eps = 1e-8)
    ?(weight_decay = 0.)
    ?(momentum = 0.)
    ?(centered = false)
    vs
    ~learning_rate
  =
  Optimizer.rmsprop ~learning_rate ~alpha ~eps ~weight_decay ~momentum ~centered
  |> create ~vs

let sgd
    ?(momentum = 0.)
    ?(dampening = 0.)
    ?(weight_decay = 0.)
    ?(nesterov = false)
    vs
    ~learning_rate
  =
  Optimizer.sgd ~learning_rate ~momentum ~dampening ~weight_decay ~nesterov |> create ~vs

let clip_grad_value_ t ~max_value =
  List.iter t.trainable_vars ~f:(fun (T tensor) ->
      Tensor.grad tensor
      |> Tensor.clamp_ ~min:(Scalar.f (-.max_value)) ~max:(Scalar.f max_value)
      |> fun tensor -> ignore (tensor : _ Tensor.t))

let clip_grad_norm2_ t ~max_norm2 =
  let total_norm =
    List.fold t.trainable_vars ~init:0. ~f:(fun acc (Tensor.T tensor) ->
        let grad = Tensor.grad tensor in
        let grad_norm =
          if Tensor.defined grad then Tensor.norm grad |> Tensor.float_value else 0.
        in
        acc +. grad_norm)
    |> Float.sqrt
  in
  let clip_coef = max_norm2 /. (1e-6 +. total_norm) in
  if Float.( < ) clip_coef 1.
  then (
    let clip_coef = Tensor.f clip_coef in
    List.iter t.trainable_vars ~f:(fun (Tensor.T tensor) ->
        let grad = Tensor.grad tensor in
        if Tensor.defined grad then ignore (Tensor.mul_ grad clip_coef : _ Tensor.t)))

let zero_grad t = Optimizer.zero_grad t.optimizer

let step ?clip_grad t =
  (match (clip_grad : Clip_grad.t option) with
  | None -> ()
  | Some (Norm2 max_norm2) -> clip_grad_norm2_ t ~max_norm2
  | Some (Value max_value) -> clip_grad_value_ t ~max_value);
  Optimizer.step t.optimizer

let backward_step ?clip_grad t ~loss =
  zero_grad t;
  Tensor.backward loss;
  step ?clip_grad t

let set_learning_rate t ~learning_rate =
  Optimizer.set_learning_rate t.optimizer learning_rate

module Linear_interpolation = struct
  type t =
    { xs : float array
    ; ys : float array
    }

  let create vs =
    if List.is_empty vs then failwith "empty knot list";
    if not (List.is_sorted vs ~compare:(fun (x1, _) (x2, _) -> Float.compare x1 x2))
    then failwith "the input knots are not sorted";
    let _ =
      List.fold vs ~init:None ~f:(fun acc (x, _) ->
          Option.iter acc ~f:(fun prev_x ->
              if Float.( = ) x prev_x then Printf.failwithf "duplicate key %f" x ());
          Some x)
    in
    let xs, ys = List.unzip vs in
    { xs = Array.of_list xs; ys = Array.of_list ys }

  let eval t x =
    (* [t] has at least one element. *)
    if Float.( <= ) x t.xs.(0)
    then t.ys.(0)
    else if Float.( <= ) t.xs.(Array.length t.xs - 1) x
    then t.ys.(Array.length t.xs - 1)
    else (
      let index =
        Array.binary_search t.xs `First_greater_than_or_equal_to x ~compare:Float.compare
      in
      let index = Option.value_exn index in
      let prev_x, prev_y = t.xs.(index - 1), t.ys.(index - 1) in
      let next_x, next_y = t.xs.(index), t.ys.(index) in
      ((prev_y *. (next_x -. x)) +. (next_y *. (x -. prev_x))) /. (next_x -. prev_x))
end
