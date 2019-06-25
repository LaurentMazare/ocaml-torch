open Base
module In_channel = Stdio.In_channel

let image_w = 28
let image_h = 28
let image_dim = image_w * image_h
let label_count = 10

let int32_be tensor ~offset =
  let get i = Tensor.(tensor.%[(Int.( + ) offset i)]) in
  get 3 + (256 * (get 2 + (256 * (get 1 + (256 * get 0)))))

let read_images filename =
  let content = Dataset_helper.read_char_tensor filename in
  let magic_number = int32_be content ~offset:0 in
  if magic_number <> 2051
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = int32_be content ~offset:4 in
  let rows = int32_be content ~offset:8 in
  let columns = int32_be content ~offset:12 in
  Tensor.narrow content ~dim:0 ~start:16 ~length:(samples * rows * columns)
  |> Tensor.to_type ~type_:(T Float)
  |> fun images ->
  Tensor.(images / f 255.) |> Tensor.view ~size:[ samples; rows * columns ]

let read_labels filename =
  let content = Dataset_helper.read_char_tensor filename in
  let magic_number = int32_be content ~offset:0 in
  if magic_number <> 2049
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = int32_be content ~offset:4 in
  Tensor.narrow content ~dim:0 ~start:8 ~length:samples |> Tensor.to_type ~type_:(T Int64)

let read_files ?(prefix = "data") () =
  let filename = Caml.Filename.concat prefix in
  { Dataset_helper.train_images = read_images (filename "train-images-idx3-ubyte")
  ; train_labels = read_labels (filename "train-labels-idx1-ubyte")
  ; test_images = read_images (filename "t10k-images-idx3-ubyte")
  ; test_labels = read_labels (filename "t10k-labels-idx1-ubyte")
  }
