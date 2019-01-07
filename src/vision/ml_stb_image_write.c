#include <assert.h>
#include <stdio.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int validate_dim(value ba, value w, value h, value comp, int byte)
{
  size_t sz = caml_ba_byte_size(Caml_ba_array_val(ba));
  size_t expected = Int_val(w) * Int_val(h) * Int_val(comp) * byte;
  return (expected <= sz);
}

CAMLprim value ml_stbi_write_png(value filename, value w, value h, value comp, value ba)
{
  CAMLparam5(filename, w, h, comp, ba);
  int result;

  if (validate_dim(ba, w, h, comp, 1))
    result = stbi_write_png(String_val(filename), Int_val(w), Int_val(h),
        Int_val(comp), Caml_ba_data_val(ba), 0);
  else
    result = 0;

  CAMLreturn(Val_int(result));
}

CAMLprim value ml_stbi_write_bmp(value filename, value w, value h, value comp, value ba)
{
  CAMLparam5(filename, w, h, comp, ba);
  int result;

  if (validate_dim(ba, w, h, comp, 1))
    result = stbi_write_bmp(String_val(filename), Int_val(w), Int_val(h),
        Int_val(comp), Caml_ba_data_val(ba));
  else
    result = 0;

  CAMLreturn(Val_int(result));
}

CAMLprim value ml_stbi_write_tga(value filename, value w, value h, value comp, value ba)
{
  CAMLparam5(filename, w, h, comp, ba);
  int result;

  if (validate_dim(ba, w, h, comp, 1))
    result = stbi_write_tga(String_val(filename), Int_val(w), Int_val(h),
        Int_val(comp), Caml_ba_data_val(ba));
  else
    result = 0;

  CAMLreturn(Val_int(result));
}

CAMLprim value ml_stbi_write_hdr(value filename, value w, value h, value comp, value ba)
{
  CAMLparam5(filename, w, h, comp, ba);
  int result;

  if (validate_dim(ba, w, h, comp, 4))
    result = stbi_write_hdr(String_val(filename), Int_val(w), Int_val(h),
        Int_val(comp), Caml_ba_data_val(ba));
  else
    result = 0;

  CAMLreturn(Val_int(result));
}

CAMLprim value ml_stbi_write_jpg_native(value filename, value w, value h, value comp, value q, value ba)
{
  CAMLparam5(filename, w, h, comp, q);
  CAMLxparam1(ba);
  int result;

  if (validate_dim(ba, w, h, comp, 1))
    result = stbi_write_jpg(String_val(filename), Int_val(w), Int_val(h),
        Int_val(comp), Caml_ba_data_val(ba), Int_val(q));
  else
    result = 0;

  CAMLreturn(Val_int(result));
}

CAMLprim value ml_stbi_write_jpg_bytecode(value * argv, int nargs)
{
    return ml_stbi_write_jpg_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}
