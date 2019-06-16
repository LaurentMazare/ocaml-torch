open Base

let sos_token = "SOS"
let eos_token = "EOS"

type t =
  { name : string
  ; word_to_index_and_count : (string, int * int) Hashtbl.t
  ; index_to_word : (int, string) Hashtbl.t
  }

let add_word t word =
  if not (String.is_empty word)
  then (
    let length = Hashtbl.length t.word_to_index_and_count in
    Hashtbl.change t.word_to_index_and_count word ~f:(function
        | Some (index, count) -> Some (index, count + 1)
        | None ->
          Hashtbl.add_exn t.index_to_word ~key:length ~data:word;
          Some (length, 1)))

let create ~name =
  let t =
    { name
    ; word_to_index_and_count = Hashtbl.create (module String)
    ; index_to_word = Hashtbl.create (module Int)
    }
  in
  add_word t sos_token;
  add_word t eos_token;
  t

let add_sentence t sentence = String.split sentence ~on:' ' |> List.iter ~f:(add_word t)
let sos_token t = Hashtbl.find_exn t.word_to_index_and_count sos_token |> fst
let eos_token t = Hashtbl.find_exn t.word_to_index_and_count eos_token |> fst
let length t = Hashtbl.length t.word_to_index_and_count
let name t = t.name
