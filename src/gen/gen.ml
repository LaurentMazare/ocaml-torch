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
    ; "conv_transpose2d_backward_out"
    ; "conv_transpose3d_backward_out"
    ; "slow_conv_transpose2d_backward_out"
    ; "slow_conv_transpose3d_backward_out"
    ; "slow_conv3d_backward_out"
    ; "normal"
    ; "_cufft_set_plan_cache_max_size"
    ; "_cufft_clear_plan_cache"
    ; "backward"
    ; "_backward"
    ; "set_data"
    ; "_amp_non_finite_check_and_unscale_"
    ; "_cummin_helper"
    ; "_cummax_helper"
    ; "retain_grad"
    ; "_validate_sparse_coo_tensor_args"
    ; "_validate_sparse_csr_tensor_args"
    ; "count_nonzero"
    ; "_assert_async"
    ; "gradient"
    ; "linalg_vector_norm"
    ; "linalg_vector_norm_out"
    ; "linalg_matrix_norm"
    ; "linalg_matrix_norm_out"
    ; "histogram"
    ; "histogram_out"
      (* Deactivate normal_out, bernoulli_out as these result in some
       ambiguous function calls. *)
    ; "normal_out"
    ; "bernoulli_out"
    ; "nested_tensor"
    ]

let no_tensor_options =
  Set.of_list
    (module String)
    [ "zeros_like"
    ; "empty_like"
    ; "full_like"
    ; "ones_like"
    ; "rand_like"
    ; "randint_like"
    ; "randn_like"
    ]

let excluded_prefixes = [ "thnn_"; "th_"; "_foreach"; "_amp_foreach"; "linalg_norm" ]
let excluded_suffixes = [ "_forward"; "_forward_out" ]
let yaml_error yaml ~msg = failwith [%string "%{msg}, %{Yaml.to_string_exn yaml}"]

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
    | Int64Option
    | Double
    | DoubleOption
    | Tensor
    | TensorOption (* Tensor.t option *)
    | IntList
    | IntListOption
    | DoubleList
    | TensorOptList
    | TensorList
    | TensorOptions (* Tensor kind and device *)
    | Scalar
    | ScalarType
    | Device
    | String

  type arg =
    { arg_name : string
    ; arg_type : arg_type
    ; default_value : string option
    }

  let ml_arg_type arg =
    match arg.arg_type with
    | Bool -> "bool"
    | Int64 -> if String.( = ) arg.arg_name "reduction" then "Reduction.t" else "int"
    | Int64Option -> "int option"
    | Double -> "float"
    | DoubleOption -> "float option"
    | Tensor -> "t"
    | TensorOption -> "t option"
    | IntList -> "int list"
    | IntListOption -> "int list option"
    | DoubleList -> "float list"
    | TensorList -> "t list"
    | TensorOptList -> "t option list"
    | TensorOptions -> "Kind.packed * Device.t"
    | Scalar -> "'a scalar"
    | ScalarType -> "Kind.packed"
    | Device -> "Device.t"
    | String -> "string"

  let named_arg arg =
    match arg.arg_name with
    | "self" | "other" | "result" | "input" | "tensor" | "tensors" -> false
    | _ -> true

  type t =
    { name : string
    ; operator_name : string
    ; overload_name : string
    ; args : arg list
    ; returns :
        (* number of tensors that are returned *)
        [ `fixed of int | `dynamic | `nothing ]
    ; kind : [ `function_ | `method_ ]
    }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some (if is_nullable then Int64Option else Int64)
    | "double" -> Some (if is_nullable then DoubleOption else Double)
    | "at::tensor" -> Some (if is_nullable then TensorOption else Tensor)
    | "at::tensoroptions" -> Some TensorOptions
    | "at::intarrayref" -> Some (if is_nullable then IntListOption else IntList)
    | "at::arrayref<double>" -> Some DoubleList
    | "const c10::list<c10::optional<at::tensor>> &" -> Some TensorOptList
    | "at::tensorlist" -> Some TensorList
    | "at::device" -> Some Device
    | "const at::scalar &" | "at::scalar" -> Some Scalar
    | "at::scalartype" -> Some ScalarType
    | "c10::string_view" -> Some String
    | _ -> None

  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
        match arg_type with
        | IntList | IntListOption ->
          Printf.sprintf "int64_t *%s_data, int %s_len" arg_name arg_name
        | DoubleList -> Printf.sprintf "double *%s_data, int %s_len" arg_name arg_name
        | TensorOptList | TensorList ->
          Printf.sprintf "tensor *%s_data, int %s_len" arg_name arg_name
        | TensorOptions -> Printf.sprintf "int %s_kind, int %s_device" arg_name arg_name
        | Int64Option -> Printf.sprintf "int64_t %s_v, int %s_null" arg_name arg_name
        | DoubleOption -> Printf.sprintf "double %s_v, int %s_null" arg_name arg_name
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
            | String -> "char *"
            | Int64Option
            | DoubleOption
            | IntList
            | IntListOption
            | DoubleList
            | TensorOptList
            | TensorList
            | TensorOptions -> assert false
          in
          Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "

  let c_args_list args =
    List.map args ~f:(fun { arg_name; arg_type; _ } ->
        match arg_type with
        | Scalar | Tensor -> "*" ^ arg_name
        | TensorOption -> [%string "(%{arg_name} ? *%{arg_name} : torch::Tensor())"]
        | Bool -> "(bool)" ^ arg_name
        | IntList -> [%string "torch::IntArrayRef(%{arg_name}_data, %{arg_name}_len)"]
        | IntListOption ->
          [%string
            "%{arg_name}_data == nullptr ? c10::nullopt : \
             c10::optional<torch::IntArrayRef>(torch::IntArrayRef(%{arg_name}_data, \
             %{arg_name}_len))"]
        | DoubleList ->
          [%string "at::ArrayRef<double>(%{arg_name}_data, %{arg_name}_len)"]
        | String -> [%string "std::string(%{arg_name})"]
        | TensorList -> [%string "of_carray_tensor(%{arg_name}_data, %{arg_name}_len)"]
        | TensorOptList ->
          Printf.sprintf "of_carray_tensor_opt(%s_data, %s_len)" arg_name arg_name
        | TensorOptions ->
          [%string
            "at::device(device_of_int(%{arg_name}_device)).dtype(at::ScalarType(%{arg_name}_kind))"]
        | Int64Option ->
          [%string
            "%{arg_name}_null ? c10::nullopt : c10::optional<int64_t>(%{arg_name}_v)"]
        | DoubleOption ->
          [%string
            "%{arg_name}_null ? c10::nullopt : c10::optional<double>(%{arg_name}_v)"]
        | ScalarType -> [%string "torch::ScalarType(%{arg_name})"]
        | Device -> [%string "device_of_int(%{arg_name})"]
        | Int64 | Double -> arg_name)
    |> String.concat ~sep:", "

  let c_call t =
    match t.kind with
    | `function_ -> [%string "torch::%{t.name}(%{c_args_list t.args})"]
    | `method_ ->
      (match t.args with
      | head :: tail -> [%string "%{head.arg_name}->%{t.name}(%{c_args_list tail})"]
      | [] ->
        failwith [%string "Method calls should have at least one argument %{t.name}"])

  let operator_name t =
    match String.lowercase t.operator_name with
    | "scatter_reduce" ->
      (* scatter_reduce is both an operator name and also obtained from the
         scatter operator when using the reduce overload. *)
      "_scatter_reduce"
    | "scatter_reduce_" -> "_scatter_reduce_"
    | other -> other

  let stubs_signature t =
    let args =
      List.concat_map t.args ~f:(fun arg ->
          match arg.arg_type with
          | Bool -> [ "int" ]
          | Int64 -> [ "int64_t" ]
          | Int64Option -> [ "int64_t"; "int" ]
          | Double -> [ "double" ]
          | DoubleOption -> [ "double"; "int" ]
          | Tensor -> [ "t" ]
          | TensorOption -> [ "t" ]
          | TensorOptions -> [ "int"; "int" ]
          | ScalarType -> [ "int" ]
          | Device -> [ "int" ]
          | IntList | IntListOption -> [ "ptr int64_t"; "int" ]
          | DoubleList -> [ "ptr double"; "int" ]
          | TensorOptList | TensorList -> [ "ptr t"; "int" ]
          | String -> [ "string" ]
          | Scalar -> [ "scalar" ])
      |> String.concat ~sep:" @-> "
    in
    match t.returns with
    | `nothing -> [%string "%{args} @-> returning void"]
    | `fixed _ -> [%string "ptr t @-> %{args} @-> returning void"]
    | `dynamic -> [%string "%{args} @-> returning (ptr t)"]

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
          [%string
            {|(List.map Int64.of_int %{name} |> CArray.of_list int64_t |> CArray.start) (List.length %{name})|}]
        | IntListOption ->
          [%string
            {|(match %{name} with | None -> from_voidp int64_t null | Some v -> List.map Int64.of_int v |> CArray.of_list int64_t |> CArray.start) (match %{name} with | None -> -1 | Some v -> List.length v)|}]
        | Int64Option ->
          [%string
            {| (match %{name} with | None -> Int64.zero | Some v -> Int64.of_int v) (match %{name} with | Some _ -> 1 | None -> 0) |}]
        | DoubleOption ->
          [%string
            {| (Option.value %{name} ~default:0.0) (match %{name} with | Some _ -> 1 | None -> 0) |}]
        | DoubleList ->
          [%string
            {|(%{name} |> CArray.of_list double |> CArray.start) (List.length %{name})|}]
        | TensorList ->
          [%string "(CArray.of_list t %{name} |> CArray.start) (List.length %{name})"]
        | TensorOptList ->
          [%string
            "(List.map (function Some x -> x | None -> null) %{name} |> CArray.of_list t \
             |> CArray.start) (List.length %{name})"]
        | Bool -> [%string "(if %{name} then 1 else 0)"]
        | ScalarType -> [%string "(Kind.packed_to_int %{name})"]
        | TensorOptions ->
          [%string "(Kind.packed_to_int (fst %{name})) (Device.to_int (snd %{name}))"]
        | Device -> [%string "(Device.to_int %{name})"]
        | Int64 ->
          if String.( = ) name "reduction"
          then "(Reduction.to_int reduction |> Int64.of_int)"
          else [%string "(Int64.of_int %{name})"]
        | TensorOption -> [%string "(match %{name} with | Some v -> v | None -> null)"]
        | Double | String | Scalar | Tensor -> name)
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
      let operator_name = Map.find_exn map "operator_name" |> extract_string in
      let overload_name = Map.find_exn map "overload_name" |> extract_string in
      let deprecated = Map.find_exn map "deprecated" |> extract_bool in
      let method_of =
        Map.find_exn map "method_of" |> extract_list |> List.map ~f:extract_string
      in
      let arguments = Map.find_exn map "arguments" |> extract_list in
      let returns =
        let is_tensor returns =
          let returns = extract_map returns in
          let return_type = Map.find_exn returns "dynamic_type" |> extract_string in
          String.( = ) return_type "at::Tensor"
        in
        let returns = Map.find_exn map "returns" |> extract_list in
        if List.is_empty returns
        then Some `nothing
        else if List.for_all returns ~f:is_tensor
        then Some (`fixed (List.length returns))
        else (
          match returns with
          | [ returns ] ->
            let return_type =
              Map.find_exn (extract_map returns) "dynamic_type" |> extract_string
            in
            if String.( = ) return_type "at::TensorList"
               || String.( = )
                    return_type
                    "dynamic_type: const c10::List<c10::optional<Tensor>> &"
            then Some `dynamic
            else None
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
                       let arg_type = Map.find_exn arg "dynamic_type" |> extract_string in
                       let is_nullable =
                         Map.find arg "is_nullable"
                         |> Option.value_map ~default:false ~f:extract_bool
                       in
                       let default_value =
                         Map.find arg "default" |> Option.map ~f:extract_string
                       in
                       match Func.arg_type_of_string arg_type ~is_nullable with
                       | Some Scalar when Option.is_some default_value && not is_nullable
                         -> None
                       | Some TensorOptions
                         when Option.is_some default_value
                              && Set.mem no_tensor_options name -> None
                       | Some arg_type -> Some { Func.arg_name; arg_type; default_value }
                       | None ->
                         if Option.is_some default_value
                         then None
                         else raise Not_a_simple_arg)
                 in
                 Some { Func.name; operator_name; overload_name; args; returns; kind }
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
              | `nothing ->
                pc "void atg_%s(%s) {" exported_name c_typed_args_list;
                pc "  PROTECT(";
                pc "    %s;" (Func.c_call func);
                pc "  )";
                pc "}";
                pc "";
                ph "void atg_%s(%s);" exported_name c_typed_args_list
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
      let funcs = Map.to_alist funcs |> List.chunks_of ~length:100 in
      List.iteri funcs ~f:(fun idx funcs ->
          p "module C%d(F: Cstubs.FOREIGN) = struct" idx;
          p "  open F";
          p "  type t = unit ptr";
          p "  let t : t typ = ptr void";
          p "  type scalar = unit ptr";
          p "  let scalar : scalar typ = ptr void";
          List.iter funcs ~f:(fun (exported_name, func) ->
              p "  let stubs_%s =" (Func.caml_name exported_name);
              p "    foreign \"atg_%s\"" exported_name;
              p "    (%s)" (Func.stubs_signature func);
              p "");
          p "end");
      p "module C(F: Cstubs.FOREIGN) = struct";
      List.iteri funcs ~f:(fun idx _funcs -> p "  include C%d(F)" idx);
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
          pi "  type t";
          pi "  type _ scalar";
          pi "";
          Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
              let caml_name = Func.caml_name exported_name in
              pm "let %s %s =" caml_name (Func.caml_args func);
              (match func.returns with
              | `nothing -> pm "  stubs_%s %s" caml_name (Func.caml_binding_args func)
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
                  (List.init ntensors ~f:(Printf.sprintf "t%d") |> String.concat ~sep:", ")
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
                | `nothing -> "unit"
                | `fixed 1 -> "t"
                | `fixed ntensors ->
                  List.init ntensors ~f:(fun _ -> "t") |> String.concat ~sep:" * "
                | `dynamic -> "t list"
              in
              pi "    %s" returns;
              pi "");
          pi "end"))

let methods =
  let c name args =
    { Func.name
    ; operator_name = name
    ; overload_name = ""
    ; args
    ; returns = `fixed 1
    ; kind = `method_
    }
  in
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
    List.map funcs ~f:(fun func -> Func.operator_name func, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
           match funcs with
           | [] -> assert false
           | [ func ] -> [ name, func ]
           | funcs ->
             let has_empty_overload =
               List.exists funcs ~f:(fun (func : Func.t) ->
                   String.is_empty func.overload_name)
             in
             List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
                 match Int.compare (String.length f1.name) (String.length f2.name) with
                 | 0 -> Int.compare (List.length f1.args) (List.length f2.args)
                 | cmp -> cmp)
             |> List.mapi ~f:(fun index (func : Func.t) ->
                    let operator_name = Func.operator_name func in
                    let overload_name = String.lowercase func.overload_name in
                    let name =
                      if String.is_empty overload_name
                         || (index = 0 && not has_empty_overload)
                      then operator_name
                      else if String.is_suffix operator_name ~suffix:"_"
                      then operator_name ^ overload_name ^ "_"
                      else operator_name ^ "_" ^ overload_name
                    in
                    name, func))
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename;
  write_stubs funcs stubs_filename;
  write_wrapper funcs wrapper_filename

let () =
  run
    ~yaml_filename:"third_party/pytorch/Declarations-v1.12.0.yaml"
    ~cpp_filename:"src/wrapper/torch_api_generated"
    ~stubs_filename:"src/stubs/torch_bindings_generated.ml"
    ~wrapper_filename:"src/wrapper/wrapper_generated"
