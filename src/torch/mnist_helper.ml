(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Base
module In_channel = Stdio.In_channel

let image_w = 28
let image_h = 28
let image_dim = image_w * image_h
let label_count = 10

let read_int32_be in_channel =
  let b1 = Option.value_exn (In_channel.input_byte in_channel) in
  let b2 = Option.value_exn (In_channel.input_byte in_channel) in
  let b3 = Option.value_exn (In_channel.input_byte in_channel) in
  let b4 = Option.value_exn (In_channel.input_byte in_channel) in
  b4 + 256 * (b3 + 256 * (b2 + 256 * b1))

let read_images filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2051
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let rows = read_int32_be in_channel in
  let columns = read_int32_be in_channel in
  let data = Tensor.zeros [samples; rows * columns] in
  for sample = 0 to samples - 1 do
    for idx = 0 to rows * columns - 1 do
      let v = Option.value_exn (In_channel.input_byte in_channel) in
      Tensor.set_float2 data sample idx Float.(of_int v / 255.)
    done;
  done;
  In_channel.close in_channel;
  data

let read_labels filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2049
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let data = Tensor.zeros ~kind:Int [samples] in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) in
    Tensor.set_int1 data sample v;
  done;
  In_channel.close in_channel;
  data

let one_hot labels =
  let nsamples =
    match Tensor.shape labels with
    | [nsamples] -> nsamples
    | [] | _::_::_ -> failwith "unexpected shape"
  in
  let one_hot = Tensor.zeros [nsamples; label_count] in
  for idx = 0 to nsamples - 1 do
    let lbl = Tensor.get_int1 labels idx in
    Tensor.set_float2 one_hot idx lbl 1.
  done;
  one_hot

type t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

let read_files
      ?(train_image_file = "data/train-images-idx3-ubyte")
      ?(train_label_file = "data/train-labels-idx1-ubyte")
      ?(test_image_file = "data/t10k-images-idx3-ubyte")
      ?(test_label_file = "data/t10k-labels-idx1-ubyte")
      ?(with_caching = false)
      ()
  =
  let with_caching ~read_f ~filename =
    if with_caching
    then
      let cached_file = filename ^ ".tensor" in
      try
        Tensor.load cached_file
      with
      | _ ->
        Stdio.eprintf
          "Cannot read from cached file %s, regenerating...\n%!"
          cached_file;
        let tensor = read_f filename in
        begin
          try
            Tensor.save tensor cached_file;
          with
          | _ ->
            Stdio.eprintf "Unable to saved cached file %s.\n%!" cached_file
        end;
        tensor
    else
      read_f filename
  in
  let read_onehot filename = read_labels filename |> one_hot in
  { train_images = with_caching ~read_f:read_images ~filename:train_image_file
  ; train_labels = with_caching ~read_f:read_onehot ~filename:train_label_file
  ; test_images = with_caching ~read_f:read_images ~filename:test_image_file
  ; test_labels = with_caching ~read_f:read_onehot ~filename:test_label_file
  }

let train_batch { train_images; train_labels; _ } ~batch_size ~batch_idx =
  let train_size = Tensor.shape train_images |> List.hd_exn in
  let start = Int.(%) (batch_size * batch_idx) (train_size - batch_size) in
  let batch_images = Tensor.narrow train_images ~dim:0 ~start ~len:batch_size in
  let batch_labels = Tensor.narrow train_labels ~dim:0 ~start ~len:batch_size in
  batch_images, batch_labels

let batch_accuracy ?samples t train_or_test ~batch_size ~predict =
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
    if samples <= start_index
    then sum_accuracy /. Float.of_int samples
    else
      let batch_size = Int.min batch_size (samples - start_index) in
      let images = Tensor.narrow images ~dim:0 ~start:start_index ~len:batch_size in
      let predicted_labels = predict images in
      let labels = Tensor.narrow labels ~dim:0 ~start:start_index ~len:batch_size in
      let batch_accuracy =
        Tensor.(sum (argmax predicted_labels = argmax labels) |> float_value)
      in
      loop (start_index + batch_size) (sum_accuracy +. batch_accuracy)
  in
  loop 0 0.
