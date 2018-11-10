open Base

type t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

let read_with_cache ~cache_file ~read =
  try
    Serialize.load_multi
      ~names:["traini"; "trainl"; "testi"; "testl"] ~filename:cache_file
    |> function
    | [ train_images; train_labels; test_images; test_labels ] ->
      { train_images; train_labels; test_images; test_labels }
    | _ -> assert false
  with
  | _ ->
    Stdio.eprintf
      "Cannot read from cached file %s, regenerating...\n%!"
      cache_file;
    let t = read () in
    begin
      try
        Serialize.save_multi ~filename:cache_file
          ~named_tensors:
            [ "traini", t.train_images; "trainl", t.train_labels
            ; "testi", t.test_images; "testl", t.test_labels
            ]
      with
      | _ ->
        Stdio.eprintf "Unable to save cached file %s.\n%!" cache_file
    end;
    t

let unexpected_shape ~shape =
  Printf.failwithf "Unexpected tensor shape (%s)"
    (List.map shape ~f:Int.to_string |> String.concat ~sep:", ") ()

let random_flip t =
  match Tensor.shape t with
  | [ batch_size; _; _; _ ] as shape ->
    let output = Tensor.zeros shape in
    for batch_index = 0 to batch_size - 1 do
      let output_view = Tensor.narrow output ~dim:0 ~start:batch_index ~length:1 in
      let t_view = Tensor.narrow t ~dim:0 ~start:batch_index ~length:1 in
      let to_copy = if Random.bool () then t_view else Tensor.flip t_view ~dims:[3] in
      Tensor.copy_ output_view ~src:to_copy
    done;
    output
  | shape -> unexpected_shape ~shape

let random_crop t ~pad =
  match Tensor.shape t with
  | [ batch_size; _; dim_h; dim_w ] as shape ->
    let padded = Tensor.reflection_pad2d t ~padding:[ pad; pad; pad; pad ] in
    let output = Tensor.zeros shape in
    for batch_index = 0 to batch_size - 1 do
      let output_view = Tensor.narrow output ~dim:0 ~start:batch_index ~length:1 in
      let start_w = Random.int (2 * pad) in
      let start_h = Random.int (2 * pad) in
      let cropped_view =
        Tensor.narrow padded ~dim:0 ~start:batch_index ~length:1
        |> Tensor.narrow ~dim:2 ~start:start_h ~length:dim_h
        |> Tensor.narrow ~dim:3 ~start:start_w ~length:dim_w
      in
      Tensor.copy_ output_view ~src:cropped_view
    done;
    output
  | shape -> unexpected_shape ~shape

let random_cutout t ~size =
  match Tensor.shape t with
  | [ batch_size; _; dim_h; dim_w ] as shape ->
    if size > dim_h || size > dim_w
    then
      Printf.failwithf "cutout %d is too large for image dimensions %d %d"
        size dim_w dim_h ();
    let output = Tensor.zeros shape in
    Tensor.copy_ output ~src:t;
    for batch_index = 0 to batch_size - 1 do
      let start_w = Random.int (dim_w - size + 1) in
      let start_h = Random.int (dim_h - size + 1) in
      Tensor.narrow output ~dim:0 ~start:batch_index ~length:1
      |> Tensor.narrow ~dim:2 ~start:start_h ~length:size
      |> Tensor.narrow ~dim:3 ~start:start_w ~length:size
      |> fun t -> Tensor.fill_float t 0.
    done;
    output
  | shape -> unexpected_shape ~shape

let train_batch ?device ?(augmentation=[]) t ~batch_size ~batch_idx =
  let { train_images; train_labels; _ } = t in
  let train_size = Tensor.shape train_images |> List.hd_exn in
  let start = Int.(%) (batch_size * batch_idx) (train_size - batch_size) in
  let batch_images = Tensor.narrow train_images ~dim:0 ~start ~length:batch_size in
  let batch_labels = Tensor.narrow train_labels ~dim:0 ~start ~length:batch_size in
  let batch_images =
    List.fold augmentation ~init:batch_images ~f:(fun batch_images augmentation ->
      match augmentation with
      | `flip -> random_flip batch_images
      | `crop_with_pad pad -> random_crop batch_images ~pad
      | `cutout size -> random_cutout batch_images ~size)
  in
  Tensor.to_device batch_images ?device, Tensor.to_device batch_labels ?device

let test_batch ?device t ~batch_size ~batch_idx =
  let { test_images; test_labels; _ } = t in
  let test_size = Tensor.shape test_images |> List.hd_exn in
  let start = Int.(%) (batch_size * batch_idx) (test_size - batch_size) in
  let batch_images = Tensor.narrow test_images ~dim:0 ~start ~length:batch_size in
  let batch_labels = Tensor.narrow test_labels ~dim:0 ~start ~length:batch_size in
  Tensor.to_device batch_images ?device, Tensor.to_device batch_labels ?device

let batch_accuracy ?device ?samples t train_or_test ~batch_size ~predict =
  let images, labels =
    match train_or_test with
    | `train -> t.train_images, t.train_labels
    | `test -> t.test_images, t.test_labels
  in
  let dataset_samples = Tensor.shape labels |> List.hd_exn in
  let samples =
    Option.value_map samples ~default:dataset_samples ~f:(Int.min dataset_samples)
  in
  let rec loop start_index sum_accuracy =
    Caml.Gc.compact ();
    if samples <= start_index
    then sum_accuracy /. Float.of_int samples
    else
      let batch_size = Int.min batch_size (samples - start_index) in
      let images =
        Tensor.narrow images ~dim:0 ~start:start_index ~length:batch_size
        |> Tensor.to_device ?device
      in
      let predicted_labels = predict images in
      let labels =
        Tensor.narrow labels ~dim:0 ~start:start_index ~length:batch_size
        |> Tensor.to_device ?device
      in
      let batch_accuracy =
        Tensor.(sum (argmax predicted_labels = labels) |> float_value)
      in
      loop (start_index + batch_size) (sum_accuracy +. batch_accuracy)
  in
  loop 0 0.

let shuffle_ t =
  let batch_size = Tensor.shape t.train_images |> List.hd_exn in
  let index = Tensor.randperm ~n:batch_size ~options:(Int64, Cpu) in
  { t with
    train_images = Tensor.index_select t.train_images ~dim:0 ~index
  ; train_labels = Tensor.index_select t.train_labels ~dim:0 ~index
  }

let batches_per_epoch t ~batch_size =
  (Tensor.shape t.train_images |> List.hd_exn) / batch_size

let test_batches_per_epoch t ~batch_size =
  (Tensor.shape t.test_images |> List.hd_exn) / batch_size

let iter ?device ?augmentation ?(shuffle = true) t ~f ~batch_size =
  let t = if shuffle then shuffle_ t else t in
  for batch_idx = 0 to batches_per_epoch t ~batch_size - 1 do
    let batch_images, batch_labels =
      train_batch ?device ?augmentation t ~batch_size ~batch_idx
    in
    f batch_idx ~batch_images ~batch_labels;
    Caml.Gc.full_major ()
  done

let map ?device t ~f ~batch_size =
  let train_images, train_labels =
    List.init (batches_per_epoch t ~batch_size) ~f:(fun batch_idx ->
      let batch_images, batch_labels = train_batch ?device t ~batch_size ~batch_idx in
      Caml.Gc.full_major ();
      f batch_idx ~batch_images ~batch_labels)
    |> List.unzip
  in
  let test_images, test_labels =
    List.init (test_batches_per_epoch t ~batch_size) ~f:(fun batch_idx ->
      let batch_images, batch_labels = test_batch ?device t ~batch_size ~batch_idx in
      Caml.Gc.full_major ();
      f batch_idx ~batch_images ~batch_labels)
    |> List.unzip
  in
  { train_images = Tensor.cat train_images ~dim:0
  ; train_labels = Tensor.cat train_labels ~dim:0
  ; test_images = Tensor.cat test_images ~dim:0
  ; test_labels = Tensor.cat test_labels ~dim:0
  }

let print_summary { train_images; train_labels; test_images; test_labels } =
  Tensor.print_shape ~name:"train-images" train_images;
  Tensor.print_shape ~name:"test-images" test_images;
  Stdio.printf "Train labels: %d, test labels: %d.\n"
    (1 + (Tensor.max_values ~dim:0 ~keepdim:false train_labels |> Tensor.int_value))
    (1 + (Tensor.max_values ~dim:0 ~keepdim:false test_labels |> Tensor.int_value))

let shuffle = shuffle_
