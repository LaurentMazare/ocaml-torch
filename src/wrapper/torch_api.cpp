#include <ATen/ATen.h>
#include<vector>
#include<caml/fail.h>
#include "torch_api.h"

#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    caml_failwith(strdup(e.what())); \
  }

using namespace std;

vector<long int> of_carray(int *vs, int len) {
  vector<long int> result;
  for (int i = 0; i < len; ++i) result.push_back(vs[i]);
  return result;
}

tensor at_zeros(int *dim_list, int dim_len, int type) {
  PROTECT(
    return new at::Tensor(at::zeros(of_carray(dim_list, dim_len), at::ScalarType(type)));
  )
}

tensor at_ones(int *dim_list, int dim_len, int type) {
  PROTECT(
    return new at::Tensor(at::ones(of_carray(dim_list, dim_len), at::ScalarType(type)));
  )
}

tensor at_rand(int *dim_list, int dim_len) {
  PROTECT(
    return new at::Tensor(at::rand(of_carray(dim_list, dim_len)));
  )
}

tensor at_reshape(tensor t, int *dim_list, int dim_len) {
  PROTECT(
    return new at::Tensor(at::reshape(*t, of_carray(dim_list, dim_len)));
  )
}

int at_dim(tensor t) {
  PROTECT(return t->dim();)
}

void at_shape(tensor t, int *dims) {
  PROTECT(
    int i = 0;
    for (int dim : t->sizes()) dims[i++] = dim;
  )
}

int at_scalar_type(tensor t) {
  PROTECT(
    return static_cast<int>(t->scalar_type());
  )
}

tensor at_add(tensor t1, tensor t2) {
  PROTECT(
    return new at::Tensor(add(*t1, *t2));
  )
}

tensor at_sub(tensor t1, tensor t2) {
  PROTECT(
    return new at::Tensor(sub(*t1, *t2));
  )
}

tensor at_mul(tensor t1, tensor t2) {
  PROTECT(
    return new at::Tensor(mul(*t1, *t2));
  )
}

tensor at_div(tensor t1, tensor t2) {
  PROTECT(
    return new at::Tensor(div(*t1, *t2));
  )
}

tensor at_pow(tensor t1, tensor t2) {
  PROTECT(
    return new at::Tensor(pow(*t1, *t2));
  )
}

tensor at_matmul(tensor t1, tensor t2) {
  PROTECT(
    return new at::Tensor(matmul(*t1, *t2));
  )
}

void at_print(tensor t) {
  PROTECT(
    at::Tensor *tensor = (at::Tensor*)t;
    cout << *tensor << endl;
  )
}

void at_free(tensor t) {
  free((at::Tensor *)t);
}
