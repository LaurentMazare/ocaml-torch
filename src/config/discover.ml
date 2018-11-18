open Base
module C = Configurator.V1

let empty_flags = { C.Pkg_config.cflags = []; libs = [] }
let combine (flags1 : C.Pkg_config.package_conf) (flags2 : C.Pkg_config.package_conf) =
  { C.Pkg_config.cflags = flags1.cflags @ flags2.cflags
  ; libs = flags1.libs @ flags2.libs
  }

let extract_flags c ~package =
  Option.bind (C.Pkg_config.get c) ~f:(C.Pkg_config.query ~package)

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
  let cflags =
    [ "-isystem" ; Printf.sprintf "%s" libroot_include
    ; "-isystem"; Printf.sprintf "%s/torch/csrc/api/include" libroot_include
    ]
  in
  let libs =
    [ Printf.sprintf "-Wl,-R%s" libroot_lib
    ; Printf.sprintf "-L%s" libroot_lib
    ; "-lc10"; "-lcaffe2"; "-ltorch"
    ]
  in
  { C.Pkg_config.cflags; libs }

let libcuda_flags ~lcuda ~lnvrtc =
  let cudadir = "/usr/local/cuda/lib64" in
  if Caml.Sys.file_exists cudadir && Caml.Sys.is_directory cudadir
  then
    let libs =
      [ Printf.sprintf "-Wl,-R%s" cudadir
      ; Printf.sprintf "-L%s" cudadir
      ]
    in
    let libs = if lcuda then libs @ [ "-lcuda" ] else libs in
    let libs = if lnvrtc then libs @ [ "-lnvrtc" ] else libs in
    { C.Pkg_config.cflags = [] ; libs }
  else empty_flags

let () =
  C.main ~name:"torch-config" (fun c ->
      let torch_flags =
        try
          torch_flags ()
        with
        | _ -> empty_flags
      in
      let cuda_flags = extract_flags c ~package:"cuda" in
      let nvrtc_flags = extract_flags c ~package:"nvrtc" in
      let cuda_flags =
        match cuda_flags, nvrtc_flags with
        | None, None -> libcuda_flags ~lcuda:true ~lnvrtc:true
        | Some cuda_flags, None ->
          combine cuda_flags (libcuda_flags ~lcuda:false ~lnvrtc:true)
        | None, Some nvrtc_flags ->
          combine nvrtc_flags (libcuda_flags ~lcuda:true ~lnvrtc:false)
        | Some cuda_flags, Some nvrtc_flags -> combine cuda_flags nvrtc_flags
      in
      C.Flags.write_sexp "cxx_flags.sexp" (torch_flags.cflags @ cuda_flags.cflags);
      C.Flags.write_sexp "c_library_flags.sexp" (torch_flags.libs @ cuda_flags.libs))
