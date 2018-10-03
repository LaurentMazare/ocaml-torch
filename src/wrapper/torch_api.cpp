#include <ATen/ATen.h>
#include<vector>
#include<caml/fail.h>
#include "torch_api.h"

using namespace std;

vector<long int> of_carray(int *vs, int len) {
  vector<long int> result;
  for (int i = 0; i < len; ++i) result.push_back(vs[i]);
  return result;
}

tensor at_zeros(int *dim_list, int dim_len) {
  try {
    return new at::Tensor(at::zeros(of_carray(dim_list, dim_len)));
  } catch (const exception& e) {
    caml_failwith(strdup(e.what()));
  }
}

tensor at_ones(int *dim_list, int dim_len) {
  try {
    return new at::Tensor(at::ones(of_carray(dim_list, dim_len)));
  } catch (const exception& e) {
    caml_failwith(strdup(e.what()));
  }
}

tensor at_rand(int *dim_list, int dim_len) {
  try {
    return new at::Tensor(at::rand(of_carray(dim_list, dim_len)));
  } catch (const exception& e) {
    caml_failwith(strdup(e.what()));
  }
}

tensor at_reshape(tensor t, int *dim_list, int dim_len) {
  try {
    return new at::Tensor(at::reshape(*t, of_carray(dim_list, dim_len)));
  } catch (const exception& e) {
    caml_failwith(strdup(e.what()));
  }
}

tensor at_add(tensor t1, tensor t2) {
  try {
    return new at::Tensor(add(*t1, *t2));
  } catch (const exception& e) {
    caml_failwith(strdup(e.what()));
  }
}

void at_print(tensor t) {
  try {
    at::Tensor *tensor = (at::Tensor*)t;
    cout << *tensor << endl;
  } catch (const exception& e) {
    caml_failwith(strdup(e.what()));
  }
}

void at_free(tensor t) {
  free((at::Tensor *)t);
}
