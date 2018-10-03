#include <ATen/Tensor.h>
#include "torch_api.h"

tensor at_zeros() {
  at::Tensor *t = new at::Tensor();
  return t;
}

tensor at_ones() {
  at::Tensor *t = new at::Tensor();
  return t;
}

tensor at_add(tensor t1, tensor t2) {
  return nullptr;
}


void at_free(tensor t) {
  free(t);
}
