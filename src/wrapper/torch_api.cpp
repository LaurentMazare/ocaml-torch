#include <torch/torch.h>
#include<vector>
#include<caml/fail.h>
#include "torch_api.h"

using namespace std;

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

tensor at_tensor_of_data(void *vs, long int *dims, int ndims, int element_size_in_bytes, int type) {
  PROTECT(
    torch::Tensor tensor = torch::zeros(torch::IntList(dims, ndims), torch::ScalarType(type));
    if (element_size_in_bytes != tensor.type().elementSizeInBytes())
      caml_failwith("incoherent element sizes in bytes");
    void *tensor_data = tensor.storage().data_ptr().get();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return new torch::Tensor(tensor);
  )
}

void at_copy_data(tensor tensor, void *vs, int64_t numel, int element_size_in_bytes) {
  PROTECT(
    if (element_size_in_bytes != tensor->type().elementSizeInBytes())
      caml_failwith("incoherent element sizes in bytes");
    if (numel != tensor->numel())
      caml_failwith("incoherent number of elements");
    void *tensor_data = tensor->storage().data_ptr().get();
    memcpy(vs, tensor_data, numel * element_size_in_bytes);
  )
}

tensor at_float_vec(double *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    return new torch::Tensor(tensor);
  )
}

tensor at_int_vec(int64_t *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    return new torch::Tensor(tensor);
  )
}

int at_defined(tensor t) {
  PROTECT(return t->defined();)
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

void at_backward(tensor t) {
  PROTECT(t->backward();)
}

int at_requires_grad(tensor t) {
  PROTECT(return t->requires_grad();)
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
}

tensor at_select(tensor t, int dim, int index) {
  PROTECT(return new torch::Tensor(select(*t, dim, index));)
}

double at_double_value(tensor t) {
  PROTECT(return t->item<double>();)
}

int64_t at_int64_value(tensor t) {
  PROTECT(return t->item<int64_t>();)
}

void at_fill_double(tensor t, double v) {
  PROTECT(t->fill_(v);)
}

void at_fill_int64(tensor t, int64_t v) {
  PROTECT(t->fill_(v);)
}

void at_set_double2(tensor t, int dim1, int dim2, double v) {
  PROTECT(
     auto dtype = t->dtype();
     if (dtype == torch::ScalarType::Float) {
       auto accessor = (t->accessor<float, 2>());
       accessor[dim1][dim2] = v;
     }
     else if (dtype == torch::ScalarType::Double) {
       auto accessor = (t->accessor<double, 2>());
       accessor[dim1][dim2] = v;
     }
     else
       caml_failwith("unexpected tensor type");
  )
}

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}

void at_save(tensor t, char *filename) {
  PROTECT(torch::save(*t, filename);)
}

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::OutputArchive archive;
    for (int i = 0; i < ntensors; ++i)
      archive.write(std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/ false);
    archive.save_to(filename);
  )
}

void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    vector<torch::Tensor> ts(ntensors);
    for (int i = 0; i < ntensors; ++i)
      archive.read(std::string(tensor_names[i]), ts[i]);
    // Only allocate the new tensor now so that if there is an exception raised during
    // [read], no memory has to be freed.
    for (int i = 0; i < ntensors; ++i)
      tensors[i] = new torch::Tensor(ts[i]);
  )
}

void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    for (int i = 0; i < ntensors; ++i)
      archive.read(std::string(tensor_names[i]), (*tensors)[i]);
  )
}

tensor at_load(char *filename) {
  PROTECT(
    torch::Tensor tensor;
    torch::load(tensor, filename);
    return new torch::Tensor(tensor);
  )
}

void at_free(tensor t) {
  delete(t);
}

optimizer ato_adam(tensor *tensors, int ntensors, double learning_rate) {
  PROTECT(
    return new torch::optim::Adam(of_carray_tensor(tensors, ntensors), learning_rate);
  )
}

void ato_zero_grad(optimizer t) {
  PROTECT(t->zero_grad();)
}

void ato_step(optimizer t) {
  PROTECT(t->step();)
}

void ato_free(optimizer t) {
  delete(t);
}

scalar ats_int(int64_t v) {
  PROTECT(return new torch::Scalar(v);)
}

scalar ats_float(double v) {
  PROTECT(return new torch::Scalar(v);)
}

void ats_free(scalar s) {
  delete(s);
}

int atc_cuda_device_count() {
  PROTECT(return torch::cuda::device_count();)
}

int atc_cuda_is_available() {
  PROTECT(return torch::cuda::is_available();)
}

int atc_cudnn_is_available() {
  PROTECT(return torch::cuda::cudnn_is_available();)
}

#include "torch_api_generated.cpp.h"
