open Base
open Torch

let npz_tensors ~filename ~f =
  let npz_file = Npy.Npz.open_in filename in
  let named_tensors =
    Npy.Npz.entries npz_file
    |> List.map ~f:(fun tensor_name ->
      f tensor_name (Npy.Npz.read npz_file tensor_name))
  in
  Npy.Npz.close_in npz_file;
  named_tensors

let ls files =
  List.iter files ~f:(fun filename ->
    Stdio.printf "%s:\n" filename;
    let tensor_names_and_shapes =
      if String.is_suffix filename ~suffix:".npz"
      then
        npz_tensors ~filename ~f:(fun tensor_name packed_tensor ->
          match packed_tensor with
          | Npy.P tensor ->
            let tensor_shape = Bigarray.Genarray.dims tensor |> Array.to_list in
            tensor_name, tensor_shape)
      else
        Serialize.load_all ~filename
        |> List.map ~f:(fun (tensor_name, tensor) ->
            tensor_name, Tensor.shape tensor)
    in
    List.iter tensor_names_and_shapes ~f:(fun (tensor_name, shape) ->
      let shape = List.map shape ~f:Int.to_string |> String.concat ~sep:", " in
      Stdio.printf "  %s (%s)\n" tensor_name shape))

let npz_to_pytorch npz_src pytorch_dst =
  let named_tensors =
    npz_tensors ~filename:npz_src ~f:(fun tensor_name packed_tensor ->
      match packed_tensor with
      | Npy.P tensor ->
        match Bigarray.Genarray.layout tensor with
        | C_layout -> tensor_name, Tensor.of_bigarray tensor
        | Fortran_layout -> failwith "fortran layout is not supported")
  in
  Serialize.save_multi ~named_tensors ~filename:pytorch_dst

let image_to_tensor image_src pytorch_dst resize =
  let resize =
    Option.map resize ~f:(fun resize ->
        match String.split_on_chars resize ~on:[ 'x'; ',' ] with
        | [ w; h ] -> Int.of_string w, Int.of_string h
        | _ -> Printf.failwithf "resize should have format WxH, e.g. 64x64" ())
  in
  let tensor =
    if Caml.Sys.is_directory image_src
    then Torch_vision.Image.load_images image_src ?resize
    else Torch_vision.Image.load_image image_src ?resize |> Or_error.ok_exn
  in
  Stdio.printf "Writing tensor with shape [%s].\n%!" (Tensor.shape_str tensor);
  Serialize.save tensor ~filename:pytorch_dst

let pytorch_to_npz pytorch_src npz_dst =
  let named_tensors = Serialize.load_all ~filename:pytorch_src in
  let npz_file = Npy.Npz.open_out npz_dst in
  List.iter named_tensors ~f:(fun (tensor_name, tensor) ->
    let write kind =
      let tensor = Tensor.to_bigarray tensor ~kind in
      Npy.Npz.write npz_file tensor_name tensor
    in
    match Tensor.kind tensor with
    | Float -> write Bigarray.float32
    | Double -> write Bigarray.float64
    | Int -> write Bigarray.int32
    | Int64 -> write Bigarray.int64
    | _ ->
      Printf.failwithf "unsupported tensor kind for %s" tensor_name ()
  );
  Npy.Npz.close_out npz_file

let () =
  let open Cmdliner in
  let ls_cmd =
    let files = Arg.(value & (pos_all file) [] & info [] ~docv:"FILE") in
    let doc = "list tensors in Npz/PyTorch files" in
    let man =
      [ `S "DESCRIPTION"
      ; `P "List all the tensors in Npz and PyTorch files."
      ]
    in
    Term.(const ls $ files),
    Term.info "ls" ~sdocs:"" ~doc ~man
  in
  let npz_to_pytorch_cmd =
    let npz_src =
      Arg.(required & pos 0 (some string) None & info [] ~docv:"SRC"
               ~doc:"Npz source file")
    in
    let pytorch_dst =
      Arg.(required & pos 1 (some string) None & info [] ~docv:"DEST"
               ~doc:"PyTorch destination file")
    in
    let doc = "convert a Npz file to PyTorch" in
    let man =
      [ `S "DESCRIPTION"
      ; `P "Convert a Npz file to a PyTorch file"
      ]
    in
    Term.(const npz_to_pytorch $ npz_src $ pytorch_dst),
    Term.info "npz-to-pytorch" ~sdocs:"" ~doc ~man
  in
  let image_to_tensor =
		let resize =
			Arg.(value & opt (some string) None & info ["resize"] ~docv:"SIZE"
         ~doc:"Resize all the images to the given size, e.g. 64x64")
		in
    let image_src =
      Arg.(required & pos 0 (some string) None & info [] ~docv:"SRC"
               ~doc:"Image source file or directory")
    in
    let pytorch_dst =
      Arg.(required & pos 1 (some string) None & info [] ~docv:"DEST"
               ~doc:"PyTorch destination file")
    in
    let doc = "convert an image file to a PyTorch Tensor" in
    let man = [ `S "DESCRIPTION" ; `P doc ] in
    Term.(const image_to_tensor $ image_src $ pytorch_dst $ resize),
    Term.info "image-to-tensor" ~sdocs:"" ~doc ~man
  in
  let pytorch_to_npz_cmd =
    let pytorch_src =
      Arg.(required & pos 0 (some string) None & info [] ~docv:"SRC"
               ~doc:"PyTorch source file")
    in
    let npz_dst =
      Arg.(required & pos 1 (some string) None & info [] ~docv:"DEST"
               ~doc:"Npz destination file")
    in
    let doc = "convert a PyTorch file to Npz" in
    let man =
      [ `S "DESCRIPTION"
      ; `P "Convert a PyTorch file to a Npz file"
      ]
    in
    Term.(const pytorch_to_npz $ pytorch_src $ npz_dst),
    Term.info "pytorch-to-npz" ~sdocs:"" ~doc ~man
  in
  let default_cmd =
    let doc = "tools for Npz tools and PyTorch archives" in
    Term.(ret (const (`Help (`Pager, None)))),
    Term.info "tensor_tools" ~version:"0.0.1" ~sdocs:"" ~doc
  in
  let cmds =
    [ ls_cmd
    ; npz_to_pytorch_cmd
    ; pytorch_to_npz_cmd
    ; image_to_tensor
    ]
  in
  Term.(exit @@ eval_choice default_cmd cmds)
