(* Automatically generate the C++ -> C -> ocaml bindings.
   This takes as input the Descriptions.yaml file that gets generated when
   building PyTorch from source.
 *)
open Base
open Stdio

let excluded_functions =
  Set.of_list
    (module String)
    [ "multi_margin_loss"
    ; "multi_margin_loss_out"
    ; "log_softmax_backward_data"
    ; "softmax_backward_data"
    ; "copy_"
    ]

let excluded_prefixes = [ "_"; "thnn_"; "th_" ]
let excluded_suffixes = [ "_forward"; "_forward_out" ]
let yaml_error yaml ~msg = Printf.failwithf "%s, %s" msg (Yaml.to_string_exn yaml) ()

let extract_bool = function
  | `Bool b -> b
  | `String "true" -> true
  | `String "false" -> false
  | yaml -> yaml_error yaml ~msg:"expected bool"

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"

let extract_map = function
  | `O map -> Map.of_alist_exn (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"

let extract_string = function
  | `String s -> s
  (* The yaml spec for torch uses n which is converted to a bool. *)
  | `Bool b -> if b then "y" else "n"
  | `Float f -> Float.to_string f
  | yaml -> yaml_error yaml ~msg:"expected string"

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Double
    | Tensor
    | TensorOption
    | IntList
    | TensorList
    | TensorOptions
    | Scalar
    | ScalarType
    | Device

  type arg =
    { arg_name : string
    ; arg_type : arg_type
    ; default_value : string option
    }

  let ml_arg_type arg =
    match arg.arg_type with
    | Bool -> "bool"
    | Int64 -> if String.( = ) arg.arg_name "reduction" then "Reduction.t" else "int"
    | Double -> "float"
    | Tensor -> "_ t"
    | TensorOption -> "_ t option"
    | IntList -> "int list"
    | TensorList -> "_ t list"
    | TensorOptions -> "Kind.packed * Device.t"
    | Scalar -> "'a scalar"
    | ScalarType -> "Kind.packed"
    | Device -> "Device.t"

  let named_arg arg =
    match arg.arg_name with
    | "self" | "other" | "result" | "input" | "tensor" | "tensors" -> false
    | _ -> true

  type t =
    { name : string
    ; args : arg list
    ; returns : (* number of tensors that are returned *)
        [ `fixed of int | `dynamic ]
    ; kind : [ `function_ | `method_ ]
    }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some Int64
    | "double" -> Some Double
    | "booltensor" | "indextensor" | "tensor" ->
      Some (if is_nullable then TensorOption else Tensor)
    | "tensoroptions" -> Some TensorOptions
    | "intarrayref" | "intlist" -> Some IntList
    | "tensorlist" -> Some TensorList
    | "device" -> Some Device
    | "scalar" -> Some Scalar
    | "scalartype" -> Some ScalarType
    | _ -> None

  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
        match arg_type with
        | IntList -> Printf.sprintf "int64_t *%s_data, int %s_len" arg_name arg_name
        | TensorList -> Printf.sprintf "tensor *%s_data, int %s_len" arg_name arg_name
        | TensorOptions -> Printf.sprintf "int %s_kind, int %s_device" arg_name arg_name
        | otherwise ->
          let simple_type_cstring =
            match otherwise with
            | Bool -> "int"
            | Int64 -> "int64_t"
            | Double -> "double"
            | Tensor -> "tensor"
            | TensorOption -> "tensor"
            | ScalarType -> "int"
            | Device -> "int"
            | Scalar -> "scalar"
            | IntList | TensorList | TensorOptions -> assert false
          in
          Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "

  let c_args_list args =
    List.map args ~f:(fun { arg_name; arg_type; _ } ->
        match arg_type with
        | Scalar | Tensor -> "*" ^ arg_name
        | TensorOption -> Printf.sprintf "(%s ? *%s : torch::Tensor())" arg_name arg_name
        | Bool -> "(bool)" ^ arg_name
        | IntList ->
          Printf.sprintf "torch::IntArrayRef(%s_data, %s_len)" arg_name arg_name
        | TensorList ->
          Printf.sprintf "of_carray_tensor(%s_data, %s_len)" arg_name arg_name
        | TensorOptions ->
          Printf.sprintf
            "at::device(device_of_int(%s_device)).dtype(at::ScalarType(%s_kind))"
            arg_name
            arg_name
        | ScalarType -> Printf.sprintf "torch::ScalarType(%s)" arg_name
        | Device -> Printf.sprintf "device_of_int(%s)" arg_name
        | _ -> arg_name)
    |> String.concat ~sep:", "

  let c_call t =
    match t.kind with
    | `function_ -> Printf.sprintf "torch::%s(%s)" t.name (c_args_list t.args)
    | `method_ ->
      (match t.args with
      | head :: tail ->
        Printf.sprintf "%s->%s(%s)" head.arg_name t.name (c_args_list tail)
      | [] ->
        Printf.failwithf "Method calls should have at least one argument %s" t.name ())

  let stubs_signature t =
    let args =
      List.concat_map t.args ~f:(fun arg ->
          match arg.arg_type with
          | Bool -> [ "int" ]
          | Int64 -> [ "int64_t" ]
          | Double -> [ "double" ]
          | Tensor -> [ "t" ]
          | TensorOption -> [ "t" ]
          | TensorOptions -> [ "int"; "int" ]
          | ScalarType -> [ "int" ]
          | Device -> [ "int" ]
          | IntList -> [ "ptr int64_t"; "int" ]
          | TensorList -> [ "ptr t"; "int" ]
          | Scalar -> [ "scalar" ])
      |> String.concat ~sep:" @-> "
    in
    match t.returns with
    | `fixed _ -> Printf.sprintf "ptr t @-> %s @-> returning void" args
    | `dynamic -> Printf.sprintf "%s @-> returning (ptr t)" args

  let replace_map =
    Map.of_alist_exn (module String) [ "end", "end_"; "to", "to_"; "t", "tr" ]

  let caml_name name =
    Map.find replace_map name |> Option.value ~default:name |> String.lowercase

  let caml_args t =
    List.map t.args ~f:(fun arg ->
        if named_arg arg then "~" ^ caml_name arg.arg_name else caml_name arg.arg_name)
    |> String.concat ~sep:" "

  let caml_binding_args t =
    List.map t.args ~f:(fun arg ->
        let name = caml_name arg.arg_name in
        match arg.arg_type with
        | IntList ->
          Printf.sprintf
            "(List.map Int64.of_int %s |> CArray.of_list int64_t |> CArray.start) \
             (List.length %s)"
            name
            name
        | TensorList ->
          Printf.sprintf
            "(CArray.of_list t %s |> CArray.start) (List.length %s)"
            name
            name
        | Bool -> Printf.sprintf "(if %s then 1 else 0)" name
        | ScalarType -> Printf.sprintf "(Kind.packed_to_int %s)" name
        | TensorOptions ->
          Printf.sprintf "(Kind.packed_to_int (fst %s)) (Device.to_int (snd %s))" name name
        | Device -> Printf.sprintf "(Device.to_int %s)" name
        | Int64 ->
          if String.( = ) name "reduction"
          then "(Reduction.to_int reduction |> Int64.of_int)"
          else Printf.sprintf "(Int64.of_int %s)" name
        | TensorOption ->
          Printf.sprintf "(match %s with | Some v -> v | None -> null)" name
        | _ -> name)
    |> String.concat ~sep:" "
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    (* Split the file to avoid Yaml.of_string_exn segfaulting. *)
    In_channel.with_file filename ~f:In_channel.input_lines
    |> List.group ~break:(fun _ l -> String.length l > 0 && Char.( = ) l.[0] '-')
    |> List.concat_map ~f:(fun lines ->
           Yaml.of_string_exn (String.concat lines ~sep:"\n") |> extract_list)
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs);
  List.filter_map funcs ~f:(fun yaml ->
      let map = extract_map yaml in
      let name = Map.find_exn map "name" |> extract_string in
      let deprecated = Map.find_exn map "deprecated" |> extract_bool in
      let method_of =
        Map.find_exn map "method_of" |> extract_list |> List.map ~f:extract_string
      in
      let arguments = Map.find_exn map "arguments" |> extract_list in
      let returns =
        let is_tensor returns =
          let returns = extract_map returns in
          let return_type = Map.find_exn returns "dynamic_type" |> extract_string in
          String.( = ) return_type "Tensor"
          || String.( = ) return_type "BoolTensor"
          || String.( = ) return_type "IndexTensor"
        in
        let returns = Map.find_exn map "returns" |> extract_list in
        if List.for_all returns ~f:is_tensor
        then Some (`fixed (List.length returns))
        else (
          match returns with
          | [ returns ] ->
            let return_type =
              Map.find_exn (extract_map returns) "dynamic_type" |> extract_string
            in
            if String.( = ) return_type "TensorList" then Some `dynamic else None
          | [] | _ :: _ :: _ -> None)
      in
      let kind =
        if List.exists method_of ~f:(String.( = ) "namespace")
        then Some `function_
        else if List.exists method_of ~f:(String.( = ) "Tensor")
        then Some `method_
        else None
      in
      if (not deprecated)
         && (not
               (List.exists excluded_prefixes ~f:(fun prefix ->
                    String.is_prefix name ~prefix)))
         && (not
               (List.exists excluded_suffixes ~f:(fun suffix ->
                    String.is_suffix name ~suffix)))
         && not (Set.mem excluded_functions name)
      then
        Option.both returns kind
        |> Option.bind ~f:(fun (returns, kind) ->
               try
                 let args =
                   List.filter_map arguments ~f:(fun arg ->
                       let arg = extract_map arg in
                       let arg_name = Map.find_exn arg "name" |> extract_string in
                       let arg_type =
                         Map.find_exn arg "dynamic_type" |> extract_string
                       in
                       let is_nullable =
                         Map.find arg "is_nullable"
                         |> Option.value_map ~default:false ~f:extract_bool
                       in
                       let default_value =
                         Map.find arg "default" |> Option.map ~f:extract_string
                       in
                       match Func.arg_type_of_string arg_type ~is_nullable with
                       | Some Scalar when Option.is_some default_value && not is_nullable
                         ->
                         None
                       | Some arg_type -> Some { Func.arg_name; arg_type; default_value }
                       | None ->
                         if Option.is_some default_value
                         then None
                         else raise Not_a_simple_arg)
                 in
                 Some { Func.name; args; returns; kind }
               with
               | Not_a_simple_arg -> None)
      else None)

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n')
    s

let write_cpp funcs filename =
  Out_channel.with_file (filename ^ ".cpp.h") ~f:(fun out_cpp ->
      Out_channel.with_file (filename ^ ".h") ~f:(fun out_h ->
          let pc s = p out_cpp s in
          let ph s = p out_h s in
          pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
          pc "";
          ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
          ph "";
          Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
              let c_typed_args_list = Func.c_typed_args_list func in
              match func.returns with
              | `dynamic ->
                pc "tensor *atg_%s(%s) {" exported_name c_typed_args_list;
                pc "  PROTECT(";
                pc "    auto outputs__ = %s;" (Func.c_call func);
                (* the returned type is a C++ vector of tensors *)
                pc "    int sz = outputs__.size();";
                pc
                  "    torch::Tensor **out__ = (torch::Tensor**)malloc((sz + 1) * \
                   sizeof(torch::Tensor*));";
                pc "    for (int i = 0; i < sz; ++i)";
                pc "      out__[i] = new torch::Tensor(outputs__[i]);";
                pc "    out__[sz] = nullptr;";
                pc "    return out__;";
                pc "  )";
                pc "}";
                pc "";
                ph "tensor *atg_%s(%s);" exported_name c_typed_args_list
              | `fixed ntensors ->
                pc "void atg_%s(tensor *out__, %s) {" exported_name c_typed_args_list;
                pc "  PROTECT(";
                pc "    auto outputs__ = %s;" (Func.c_call func);
                if ntensors = 1
                then pc "    out__[0] = new torch::Tensor(outputs__);"
                else
                  for i = 0 to ntensors - 1 do
                    pc "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i
                  done;
                pc "  )";
                pc "}";
                pc "";
                ph "void atg_%s(tensor *, %s);" exported_name c_typed_args_list)))

let write_stubs funcs filename =
  Out_channel.with_file filename ~f:(fun out_channel ->
      let p s = p out_channel s in
      p "(* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)";
      p "";
      p "open Ctypes";
      p "";
      p "module C(F: Cstubs.FOREIGN) = struct";
      p "  open F";
      p "  type t = unit ptr";
      p "  let t : t typ = ptr void";
      p "  type scalar = unit ptr";
      p "  let scalar : scalar typ = ptr void";
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
          p "  let stubs_%s =" (Func.caml_name exported_name);
          p "    foreign \"atg_%s\"" exported_name;
          p "    (%s)" (Func.stubs_signature func);
          p "");
      p "end")

let write_wrapper funcs filename =
  Out_channel.with_file (filename ^ ".ml") ~f:(fun out_ml ->
      Out_channel.with_file (filename ^ "_intf.ml") ~f:(fun out_intf ->
          let pm s = p out_ml s in
          let pi s = p out_intf s in
          pm "(* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)";
          pm "";
          pm "open Ctypes";
          pm "";
          pm "module C = Torch_bindings.C(Torch_generated)";
          pm "open C.TensorG";
          pm "";
          pm "let to_tensor_list ptr =";
          pm "  let rec loop ptr acc =";
          pm "    let tensor = !@ptr in";
          pm "    if is_null tensor";
          pm "    then acc";
          pm "    else begin";
          pm "      Gc.finalise C.Tensor.free tensor;";
          pm "      loop (ptr +@ 1) (tensor :: acc)";
          pm "    end";
          pm "  in";
          pm "  let result = loop ptr [] in";
          pm "  C.free (to_voidp ptr);";
          pm "  List.rev result";
          pm "";
          pi "(* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)";
          pi "";
          pi "module type S = sig";
          pi "  type _ t";
          pi "  type _ scalar";
          pi "";
          Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
              let caml_name = Func.caml_name exported_name in
              pm "let %s %s =" caml_name (Func.caml_args func);
              (match func.returns with
              | `fixed ntensors ->
                pm "  let out__ = CArray.make t %d in" ntensors;
                pm
                  "  stubs_%s (CArray.start out__) %s;"
                  caml_name
                  (Func.caml_binding_args func);
                for i = 0 to ntensors - 1 do
                  pm "  let t%d = CArray.get out__ %d in" i i;
                  pm "  Gc.finalise C.Tensor.free t%d;" i
                done;
                pm
                  "  %s"
                  (List.init ntensors ~f:(Printf.sprintf "t%d")
                  |> String.concat ~sep:", ")
              | `dynamic ->
                pm
                  "  stubs_%s %s |> to_tensor_list"
                  caml_name
                  (Func.caml_binding_args func));
              pm "";
              pi "  val %s :" caml_name;
              List.iter func.args ~f:(fun arg ->
                  let named_arg =
                    if Func.named_arg arg
                    then Printf.sprintf "%s:" (Func.caml_name arg.arg_name)
                    else ""
                  in
                  pi "    %s%s ->" named_arg (Func.ml_arg_type arg));
              let returns =
                match func.returns with
                | `fixed 1 -> "_ t"
                | `fixed ntensors ->
                  List.init ntensors ~f:(fun _ -> "_ t") |> String.concat ~sep:" * "
                | `dynamic -> "_ t list"
              in
              pi "    %s" returns;
              pi "");
          pi "end"))

let methods =
  let c name args = { Func.name; args; returns = `fixed 1; kind = `method_ } in
  let ca arg_name arg_type = { Func.arg_name; arg_type; default_value = None } in
  [ c "grad" [ ca "self" Tensor ]
  ; c "set_requires_grad" [ ca "self" Tensor; ca "r" Bool ]
  ; c "toType" [ ca "self" Tensor; ca "scalar_type" ScalarType ]
  ; c "to" [ ca "self" Tensor; ca "device" Device ]
  ]

let run ~yaml_filename ~cpp_filename ~stubs_filename ~wrapper_filename =
  let funcs = read_yaml yaml_filename in
  let funcs = methods @ funcs in
  printf "Generating code for %d functions.\n%!" (List.length funcs);
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> String.lowercase func.name, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
           match funcs with
           | [] -> assert false
           | [ func ] -> [ name, func ]
           | funcs ->
             List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
                 Int.compare (List.length f1.args) (List.length f2.args))
             |> List.mapi ~f:(fun i func ->
                    (if i = 0 then name else Printf.sprintf "%s%d" name i), func))
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename;
  write_stubs funcs stubs_filename;
  write_wrapper funcs wrapper_filename

let () =
  run
    ~yaml_filename:"data/Declarations.yaml"
    ~cpp_filename:"src/wrapper/torch_api_generated"
    ~stubs_filename:"src/stubs/torch_bindings_generated.ml"
    ~wrapper_filename:"src/wrapper/wrapper_generated"
