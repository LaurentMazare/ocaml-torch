open Base
open Torch

type buffer = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external resize_
  :  in_data:buffer
  -> in_w:int
  -> in_h:int
  -> out_data:buffer
  -> out_w:int
  -> out_h:int
  -> nchannels:int
  -> int
  = "ml_stbir_resize_bytecode" "ml_stbir_resize"

let tensor_of_data ~data ~width ~height =
  Tensor.of_bigarray (Bigarray.genarray_of_array1 data)
  |> Tensor.view ~size:[ 1; height; width; 3 ]
  |> Tensor.permute ~dims:[ 0; 3; 1; 2 ]

let maybe_crop tensor ~dim ~length ~target_length =
  if length < target_length
  then
    Printf.failwithf "target_length greater than length %d > %d" target_length length ();
  if length = target_length
  then tensor
  else
    Tensor.narrow tensor ~dim ~start:((length - target_length) / 2) ~length:target_length

let load_image ?resize image_file =
  Stb_image.load image_file
  |> Result.bind ~f:(fun (image : _ Stb_image.t) ->
         if image.channels = 3
         then
           (match resize with
           | None ->
             tensor_of_data ~data:image.data ~width:image.width ~height:image.height
           | Some (target_width, target_height) ->
             (* First resize the image while preserving the ratio. *)
             let resize_width, resize_height =
               let ratio_w = Float.of_int target_width /. Float.of_int image.width in
               let ratio_h = Float.of_int target_height /. Float.of_int image.height in
               let r = Float.max ratio_w ratio_h in
               ( Float.to_int (r *. Float.of_int image.width) |> Int.max target_width
               , Float.to_int (r *. Float.of_int image.height) |> Int.max target_height )
             in
             let out_data =
               Bigarray.Array1.create
                 Int8_unsigned
                 C_layout
                 (resize_width * resize_height * 3)
             in
             let status =
               resize_
                 ~in_data:image.data
                 ~in_w:image.width
                 ~in_h:image.height
                 ~out_data
                 ~out_w:resize_width
                 ~out_h:resize_height
                 ~nchannels:3
             in
             if status = 0 then Printf.failwithf "error when resizing %s" image_file ();
             let tensor =
               tensor_of_data ~data:out_data ~width:resize_width ~height:resize_height
             in
             (* Then take a center crop. *)
             maybe_crop tensor ~dim:3 ~length:resize_width ~target_length:target_width
             |> maybe_crop ~dim:2 ~length:resize_height ~target_length:target_height)
           |> Result.return
         else Error (`Msg (Printf.sprintf "%d channels <> 3" image.channels)))
  |> Result.map_error ~f:(fun (`Msg msg) -> Error.of_string msg)

let image_suffixes = [ ".jpg"; ".png" ]

let has_image_suffix filename =
  List.exists image_suffixes ~f:(fun suffix -> String.is_suffix filename ~suffix)

let load_images ?resize dir =
  if not (Caml.Sys.is_directory dir) then Printf.failwithf "not a directory %s" dir ();
  let files = Caml.Sys.readdir dir |> Array.to_list in
  Stdio.printf "%d files found in %s\n%!" (List.length files) dir;
  List.filter_map files ~f:(fun filename ->
      if has_image_suffix filename
      then (
        match load_image (Caml.Filename.concat dir filename) ?resize with
        | Ok image -> Some image
        | Error msg ->
          Stdio.printf "%s: %s\n%!" filename (Error.to_string_hum msg);
          None)
      else None)
  |> Tensor.cat ~dim:0

let load_dataset ~dir ~classes ~with_cache ~resize =
  let read () =
    let load tv s = load_images (Printf.sprintf "%s/%s/%s" dir tv s) ~resize in
    let load_tv tv =
      List.mapi classes ~f:(fun class_index class_dir ->
          let images = load tv class_dir in
          let labels =
            Tensor.zeros [ Tensor.shape images |> List.hd_exn ] ~kind:(T Int64)
          in
          Tensor.fill_int labels class_index;
          images, labels)
      |> List.unzip
      |> fun (images, labels) -> Tensor.cat images ~dim:0, Tensor.cat labels ~dim:0
    in
    let train_images, train_labels = load_tv "train" in
    let test_images, test_labels = load_tv "val" in
    { Dataset_helper.train_images; train_labels; test_images; test_labels }
  in
  match with_cache with
  | None -> read ()
  | Some cache_file -> Dataset_helper.read_with_cache ~cache_file ~read

let write_image tensor ~filename =
  let tensor, height, width, channels, chw =
    match Tensor.shape tensor with
    | [ 1; channels; b; c ] when channels = 1 || channels = 3 ->
      Tensor.reshape tensor ~shape:[ channels; b; c ], b, c, channels, true
    | [ channels; b; c ] when channels = 1 || channels = 3 ->
      tensor, b, c, channels, true
    | [ b; c; channels ] when channels = 1 || channels = 3 ->
      tensor, b, c, channels, false
    | _ -> Printf.failwithf "unexpected shape %s" (Tensor.shape_str tensor) ()
  in
  let bigarray =
    let tensor =
      if chw
      then Tensor.permute tensor ~dims:[ 1; 2; 0 ] |> Tensor.contiguous
      else tensor
    in
    Tensor.view tensor ~size:[ channels * height * width ]
    |> Tensor.to_type ~type_:Uint8
    |> Tensor.to_bigarray ~kind:Int8_unsigned
    |> Bigarray.array1_of_genarray
  in
  match String.rsplit2 filename ~on:'.' with
  | Some (_, "jpg") ->
    Stb_image_write.jpg filename bigarray ~w:width ~h:height ~c:channels ~quality:90
  | Some (_, "tga") ->
    Stb_image_write.tga filename bigarray ~w:width ~h:height ~c:channels
  | Some (_, "bmp") ->
    Stb_image_write.bmp filename bigarray ~w:width ~h:height ~c:channels
  | Some (_, "png") ->
    Stb_image_write.png filename bigarray ~w:width ~h:height ~c:channels
  | Some _ | None ->
    Stb_image_write.png (filename ^ ".png") bigarray ~w:width ~h:height ~c:channels

(* TODO: think about how to do a multi-threaded version of this. *)
module Loader = struct
  type t =
    { dir_and_filenames : (string * string) array
    ; mutable current_index : int
    ; resize : (int * int) option
    }

  let nsamples t = Array.length t.dir_and_filenames

  let create ?resize ~dir () =
    if not (Caml.Sys.is_directory dir) then Printf.failwithf "not a directory %s" dir ();
    let dir_and_filenames = ref [] in
    let rec walk_dir dir =
      Caml.Sys.readdir dir
      |> Array.iter ~f:(fun file ->
             let next_dir = Caml.Filename.concat dir file in
             if Caml.Sys.is_directory next_dir
             then walk_dir next_dir
             else if has_image_suffix file
             then dir_and_filenames := (dir, file) :: !dir_and_filenames)
    in
    walk_dir dir;
    { dir_and_filenames = Array.of_list !dir_and_filenames; current_index = 0; resize }

  let reset t = t.current_index <- 0

  let next_batch t ~batch_size =
    let nsamples = nsamples t in
    let batch_size = Int.min batch_size (nsamples - t.current_index) in
    if batch_size = 0
    then None
    else (
      let batch =
        List.init batch_size ~f:(fun i ->
            let dir, file = t.dir_and_filenames.(t.current_index + i) in
            load_image ?resize:t.resize (Caml.Filename.concat dir file)
            |> Or_error.ok_exn)
        |> Tensor.cat ~dim:0
      in
      t.current_index <- t.current_index + batch_size;
      Some batch)

  let random_batch t ~batch_size =
    let nsamples = nsamples t in
    List.init batch_size ~f:(fun _ ->
        let dir, file = t.dir_and_filenames.(Random.int nsamples) in
        load_image ?resize:t.resize (Caml.Filename.concat dir file) |> Or_error.ok_exn)
    |> Tensor.cat ~dim:0
end

(* Resize an image, this does not preserve the aspect ratio. *)
let resize tensor ~height ~width =
  let resize_one tensor =
    match Tensor.shape tensor with
    | [ c; h; w ] when c = 1 || c = 3 ->
      let in_data =
        Tensor.permute tensor ~dims:[ 1; 2; 0 ]
        |> Tensor.contiguous
        |> Tensor.view ~size:[ c * h * w ]
        |> Tensor.to_type ~type_:Uint8
        |> Tensor.to_bigarray ~kind:Int8_unsigned
        |> Bigarray.array1_of_genarray
      in
      let out_data =
        Bigarray.Array1.create Int8_unsigned C_layout (width * height * c)
      in
      let status =
        resize_
          ~in_data
          ~in_w:w
          ~in_h:h
          ~out_data
          ~out_w:width
          ~out_h:height
          ~nchannels:c
      in
      if status = 0 then failwith "error when resizing image";
      tensor_of_data ~data:out_data ~width ~height
    | _ -> assert false
  in
  match Tensor.shape tensor with
  | [ c; _; _ ] when c = 1 || c = 3 -> resize_one tensor
  | [ _; c; _; _ ] when c = 1 || c = 3 ->
    Tensor.to_list tensor |> List.map ~f:resize_one |> Tensor.cat ~dim:0
  | _ -> Printf.failwithf "unexpected shape %s" (Tensor.shape_str tensor) ()
