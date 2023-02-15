open Base
module C = Configurator.V1

let empty_flags = { C.Pkg_config.cflags = []; libs = [] }

let combine (flags1 : C.Pkg_config.package_conf) (flags2 : C.Pkg_config.package_conf) =
  { C.Pkg_config.cflags = flags1.cflags @ flags2.cflags
  ; libs = flags1.libs @ flags2.libs
  }

let ( /^ ) = Caml.Filename.concat
let file_exists = Caml.Sys.file_exists

let extract_flags c ~package =
  Option.bind (C.Pkg_config.get c) ~f:(C.Pkg_config.query ~package)

let torch_flags () =
  let config ~include_dir ~lib_dir =
    let cflags =
      [ "-isystem"
      ; Printf.sprintf "%s" include_dir
      ; "-isystem"
      ; Printf.sprintf "%s/torch/csrc/api/include" include_dir
      ]
    in
    let libs =
      [ Printf.sprintf "-Wl,-rpath,%s" lib_dir
      ; Printf.sprintf "-L%s" lib_dir
      ; "-lc10"
      ; "-ltorch_cpu"
      ; "-ltorch"
      ]
    in
    { C.Pkg_config.cflags; libs }
  in
  let conda_config ~conda_prefix =
    let conda_prefix = conda_prefix ^ "/lib" in
    Caml.Sys.readdir conda_prefix
    |> Array.to_list
    |> List.filter_map ~f:(fun filename ->
           if String.is_prefix filename ~prefix:"python"
           then (
             let libdir =
               Printf.sprintf "%s/%s/site-packages/torch" conda_prefix filename
             in
             if file_exists libdir && Caml.Sys.is_directory libdir
             then Some libdir
             else None)
           else None)
    |> function
    | [] -> None
    | lib_dir :: _ ->
      Some (config ~include_dir:(lib_dir /^ "include") ~lib_dir:(lib_dir /^ "lib"))
  in
  match Caml.Sys.getenv_opt "LIBTORCH" with
  | Some l -> config ~include_dir:(l /^ "include") ~lib_dir:(l /^ "lib")
  | None ->
    let conda_flags =
      Option.bind (Caml.Sys.getenv_opt "CONDA_PREFIX") ~f:(fun conda_prefix ->
          conda_config ~conda_prefix)
    in
    (match conda_flags with
    | Some conda_flags -> conda_flags
    | None ->
      (match Caml.Sys.getenv_opt "OPAM_SWITCH_PREFIX" with
      | Some prefix ->
        let lib_dir = prefix /^ "lib" /^ "libtorch" in
        if file_exists lib_dir
        then config ~include_dir:(lib_dir ^ "/include") ~lib_dir:(lib_dir ^ "/lib")
        else empty_flags
      | None -> empty_flags))

let libcuda_flags ~lcuda ~lnvrtc =
  let cudadir = "/usr/local/cuda/lib64" in
  if file_exists cudadir && Caml.Sys.is_directory cudadir
  then (
    let libs =
      [ Printf.sprintf "-Wl,-rpath,%s" cudadir; Printf.sprintf "-L%s" cudadir ]
    in
    let libs = if lcuda then libs @ [ "-lcudart" ] else libs in
    let libs = if lnvrtc then libs @ [ "-lnvrtc" ] else libs in
    { C.Pkg_config.cflags = []; libs })
  else empty_flags

let () =
  C.main ~name:"torch-config" (fun c ->
      let torch_flags =
        try torch_flags () with
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
      let conda_libs =
        Option.value_map
          (Caml.Sys.getenv_opt "CONDA_PREFIX")
          ~f:(fun conda_prefix -> [ Printf.sprintf "-Wl,-rpath,%s/lib" conda_prefix ])
          ~default:[]
      in
      let cxx_abi_flag =
        let cxx_abi =
          match Caml.Sys.getenv_opt "LIBTORCH_CXX11_ABI" with
          | Some v -> v
          | None -> "1"
        in
        Printf.sprintf "-D_GLIBCXX_USE_CXX11_ABI=%s" cxx_abi
      in
      C.Flags.write_sexp
        "cxx_flags.sexp"
        (cxx_abi_flag :: (torch_flags.cflags @ cuda_flags.cflags));
      let torch_flags_lib =
        if Caml.( = ) cuda_flags empty_flags
        then torch_flags.libs
        else "-Wl,--no-as-needed" :: torch_flags.libs
      in
      let macosx_flags_lib =
        (* The '-Wl,-keep_dwarf_unwind' flag is useful for c++ exceptions to properly be converted to ocaml
           exceptions.
           https://github.com/LaurentMazare/ocaml-torch/issues/78 *)
        match C.ocaml_config_var c "system" with
        | Some "macosx" -> [ "-Wl,-keep_dwarf_unwind" ]
        | _ -> []
      in
      C.Flags.write_sexp
        "c_library_flags.sexp"
        (macosx_flags_lib @ torch_flags_lib @ conda_libs @ cuda_flags.libs))
