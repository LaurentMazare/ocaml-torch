open Base

let one_hot labels ~label_count =
  let nsamples = Bigarray.Array1.dim labels in
  let one_hot = Bigarray.Array2.create Float32 C_layout nsamples label_count in
  Bigarray.Array2.fill one_hot 0.;
  for idx = 0 to nsamples - 1 do
    one_hot.{ idx, labels.{ idx } } <- 1.
  done;
  Bigarray.genarray_of_array2 one_hot |> Tensor.of_bigarray

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

let train_batch ?device { train_images; train_labels; _ } ~batch_size ~batch_idx =
  let train_size = Tensor.shape train_images |> List.hd_exn in
  let start = Int.(%) (batch_size * batch_idx) (train_size - batch_size) in
  let batch_images = Tensor.narrow train_images ~dim:0 ~start ~len:batch_size in
  let batch_labels = Tensor.narrow train_labels ~dim:0 ~start ~len:batch_size in
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
        Tensor.narrow images ~dim:0 ~start:start_index ~len:batch_size
        |> Tensor.to_device ?device
      in
      let predicted_labels = predict images in
      let labels =
        Tensor.narrow labels ~dim:0 ~start:start_index ~len:batch_size
        |> Tensor.to_device ?device
      in
      let batch_accuracy =
        Tensor.(sum (argmax predicted_labels = argmax labels) |> float_value)
      in
      loop (start_index + batch_size) (sum_accuracy +. batch_accuracy)
  in
  loop 0 0.
