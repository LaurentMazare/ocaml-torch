#include <ATen/ATen.h>
#include "torch_api.h"

tensor at_zeros() {
  at::Tensor *t = new at::Tensor(at::zeros({4, 2}));
  return t;
}

tensor at_ones() {
  at::Tensor *t = new at::Tensor(at::ones({4, 2}));
  return t;
}

tensor at_add(tensor t1, tensor t2) {
  at::Tensor *tensor1 = (at::Tensor*)t1;
  at::Tensor *tensor2 = (at::Tensor*)t2;
  return new at::Tensor(at::add(*tensor1, *tensor2));
}

void at_print(tensor t) {
  at::Tensor *tensor = (at::Tensor*)t;
  tensor->print();
}

void at_free(tensor t) {
  free((at::Tensor *)t);
}
