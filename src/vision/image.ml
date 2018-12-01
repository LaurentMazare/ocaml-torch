open Base
open Torch

let resize_and_crop image_file ~f ~width ~height =
  let tmp_file = Caml.Filename.temp_file "imagenet-resize-crop" ".jpg" in
  let args =
    [ "-resize"; Printf.sprintf "%dx%d^" width height
    ; "-gravity"; "center"
    ; "-crop"; Printf.sprintf "%dx%d+0+0" width height
    ; Printf.sprintf "\"%s\"" image_file; tmp_file
    ]
  in
  let command = Printf.sprintf "convert %s" (String.concat args ~sep:" ") in
  begin
    match Unix.system command with
    | WEXITED 0 -> ()
    | WEXITED i ->
      Printf.failwithf "%s returns a non-zero exit code %d" command i ()
    | WSIGNALED i -> Printf.failwithf "%s killed by signal %d" command i ()
    | WSTOPPED i -> Printf.failwithf "%s stopped %d" command i ()
  end;
  Exn.protect
    ~f:(fun () -> f tmp_file)
    ~finally:(fun () -> Unix.unlink tmp_file)

let load_image_no_resize_and_crop image_file =
  let image = ImageLib.openfile image_file in
  Stdio.printf "%s: %dx%d (%d)\n%!" image_file image.width image.height image.max_val;
  match image.pixels with
  | RGB (Pix8 red, Pix8 green, Pix8 blue) ->
    let convert pixels =
      Bigarray.genarray_of_array2 pixels
      |> Tensor.of_bigarray
    in
    let image =
      Tensor.stack [ convert red; convert green; convert blue ] ~dim:0
      |> Tensor.transpose ~dim0:1 ~dim1:2
    in
    Tensor.view image ~size:(1 :: Tensor.shape image)
  | _ -> failwith "unexpected pixmaps"

let load_image ?resize image_file =
  match resize with
  | Some (width, height) ->
    resize_and_crop image_file ~f:load_image_no_resize_and_crop ~width ~height
  | None -> load_image_no_resize_and_crop image_file

let image_suffixes = [ ".jpg"; ".png" ]

let load_images ~dir ~resize =
  if not (Caml.Sys.is_directory dir)
  then Printf.failwithf "not a directory %s" dir ();
  Caml.Sys.readdir dir
  |> Array.to_list
  |> List.filter_map ~f:(fun filename ->
    if List.exists image_suffixes ~f:(fun suffix -> String.is_suffix filename ~suffix)
    then begin
      Stdio.printf "<%s>\n%!" filename;
      try Some (load_image (Caml.Filename.concat dir filename) ~resize)
      with _ -> None
    end else None)
  |> Tensor.cat ~dim:0

let load_dataset ~dir ~classes ~with_cache ~resize =
  let read () =
    let load tv s = load_images ~dir:(Printf.sprintf "%s/%s/%s" dir tv s) ~resize in
    let load_tv tv =
      List.mapi classes ~f:(fun class_index class_dir ->
        let images = load tv class_dir in
        let labels = Tensor.zeros [ Tensor.shape images |> List.hd_exn ] ~kind:Int64 in
        Tensor.fill_int labels class_index;
        images, labels)
      |> List.unzip
      |> fun (images, labels) ->
      Tensor.cat images ~dim:0, Tensor.cat labels ~dim:0
    in
    let train_images, train_labels = load_tv "train" in
    let test_images, test_labels = load_tv "val" in
    { Dataset_helper.train_images
    ; train_labels
    ; test_images
    ; test_labels
    }
  in
  match with_cache with
  | None -> read ()
  | Some cache_file -> Dataset_helper.read_with_cache ~cache_file ~read

let write_image tensor ~filename =
  let tensor, height, width =
    match Tensor.shape tensor with
    | [ 1; a; b; c ] -> Tensor.reshape tensor ~shape:[ a; b; c ], b, c
    | [ _; b; c ] -> tensor, b, c
    | shape ->
      Printf.failwithf "unexpected shape %s"
        (List.map shape ~f:Int.to_string |> String.concat ~sep:", ") ()
  in
  let tensor = Tensor.transpose tensor ~dim0:1 ~dim1:2 in
  let extract index_ =
    Tensor.get tensor index_
    |> Tensor.to_bigarray ~kind:Int8_unsigned
    |> Bigarray.array2_of_genarray
  in
  let red = extract 0 in
  let green = extract 1 in
  let blue = extract 2 in
  ImageLib.writefile filename
    { width
    ; height
    ; max_val = 255
    ; pixels = RGB (Pix8 red, Pix8 green, Pix8 blue)
    }
