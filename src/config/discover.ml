open Base
module C = Configurator.V1

let extract_flags c ~package =
  match C.Pkg_config.get c with
  | None -> [], []
  | Some pc ->
    let cuda_cxx_flags, cuda_c_library_flags =
      match C.Pkg_config.query pc ~package with
      | None -> [], []
      | Some deps -> deps.cflags, deps.libs
    in
    cuda_cxx_flags, cuda_c_library_flags

let torch_flags () =
  let libroot_include, libroot_lib =
    try
      let libtorch = Caml.Sys.getenv "LIBTORCH" in
      if String.is_empty libtorch
      then raise Caml.Not_found;
      libtorch ^ "/include", libtorch ^ "/lib"
    with
    | _ ->
      let conda_prefix = Caml.Sys.getenv "CONDA_PREFIX" in
      let conda_prefix = conda_prefix ^ "/lib" in
      Caml.Sys.readdir conda_prefix
      |> Array.to_list
      |> List.filter_map ~f:(fun filename ->
          if String.is_prefix filename ~prefix:"python"
          then
            let libdir =
              Printf.sprintf "%s/%s/site-packages/torch/lib" conda_prefix filename
            in
            if Caml.Sys.file_exists libdir && Caml.Sys.is_directory libdir
            then Some libdir
            else None
          else None)
      |> function
      | [] -> Printf.failwithf "no python directory with torch found in %s" conda_prefix ()
      | libdir :: _ -> libdir ^ "/include", libdir
  in
  let cxx_flags =
    [ "-isystem" ; Printf.sprintf "%s" libroot_include
    ; "-isystem"; Printf.sprintf "%s/torch/csrc/api/include" libroot_include
    ]
  in
  let c_library_flags =
    [ Printf.sprintf "-Wl,-R%s" libroot_lib
    ; Printf.sprintf "-L%s" libroot_lib
    ; "-lc10"; "-lcaffe2"; "-ltorch"
    ]
  in
  cxx_flags, c_library_flags

let () =
  C.main ~name:"torch-config" (fun c ->
      let cxx_flags, c_library_flags =
        try
          torch_flags ()
        with
        | _ -> [], []
      in
      let cuda_cxx_flags, cuda_c_library_flags = extract_flags c ~package:"cuda" in
      let nvrtc_cxx_flags, nvrtc_c_library_flags = extract_flags c ~package:"nvrtc" in
      C.Flags.write_sexp "cxx_flags.sexp"
        (cxx_flags @ cuda_cxx_flags @ nvrtc_cxx_flags);
      C.Flags.write_sexp "c_library_flags.sexp"
        (c_library_flags @ cuda_c_library_flags @ nvrtc_c_library_flags))
