#include <ATen/ATen.h>
#include<vector>
#include "torch_api.h"

using namespace std;

tensor at_zeros(int *dim_list, int dim_len) {
  vector<long int> dims;
  for (int i = 0; i < dim_len; ++i)
    dims.push_back(dim_list[i]);
  return new at::Tensor(at::zeros(at::IntList(dims)));
}

tensor at_ones(int *dim_list, int dim_len) {
  vector<long int> dims;
  for (int i = 0; i < dim_len; ++i)
    dims.push_back(dim_list[i]);
  return new at::Tensor(at::ones(at::IntList(dims)));
}

tensor at_rand(int *dim_list, int dim_len) {
  vector<long int> dims;
  for (int i = 0; i < dim_len; ++i)
    dims.push_back(dim_list[i]);
  return new at::Tensor(at::rand(at::IntList(dims)));
}

tensor at_add(tensor t1, tensor t2) {
  return new at::Tensor(t1->add(*t2));
}

void at_print(tensor t) {
  at::Tensor *tensor = (at::Tensor*)t;
  cout << *tensor << endl;
}

void at_free(tensor t) {
  free((at::Tensor *)t);
}
