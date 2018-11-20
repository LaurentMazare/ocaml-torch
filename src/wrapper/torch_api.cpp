#include<torch/torch.h>
#include<torch/script.h>
#include<vector>
#include<caml/fail.h>
#include "torch_api.h"

using namespace std;

void at_manual_seed(int64_t seed) {
  torch::manual_seed(seed);
}

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

tensor at_new_tensor() {
  PROTECT(
    return new torch::Tensor();
  )
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

int at_grad_set_enabled(int b) {
  PROTECT(
    bool is_enabled = torch::autograd::GradMode::is_enabled();
    torch::autograd::GradMode::set_enabled(b);
    return is_enabled;
  )
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
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

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}

char *at_to_string(tensor t, int line_size) {
  PROTECT(
    std::ostringstream oss;
    torch::print(oss, *t, line_size);
    return strdup(oss.str().c_str());
  )
}

void at_copy_(tensor dst, tensor src) {
  PROTECT(
    dst->copy_(*src);
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

void at_load_callback(char *filename, void (*f)(char *, tensor)) {
  PROTECT(
    shared_ptr<torch::jit::script::Module> module = torch::jit::load(filename);
    if (module == nullptr)
      caml_failwith("torch::jit::load returned a nullptr");
    for (const auto &p : module->get_parameters()) {
      f((char*)p.key().c_str(), new torch::Tensor(*p.value().slot()));
    }
  )
}

void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    for (int i = 0; i < ntensors; ++i) {
      if (tensors[i]->device().type() == at::kCPU)
        archive.read(std::string(tensor_names[i]), *(tensors[i]));
      else {
        torch::Tensor tmp_tensor = torch::empty_like(*(tensors[i]), at::device(at::kCPU));
        archive.read(std::string(tensor_names[i]), tmp_tensor);
        tensors[i]->copy_(tmp_tensor);
      }
    }
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

optimizer ato_adam(double learning_rate) {
  PROTECT(
    return new torch::optim::Adam(vector<torch::Tensor>(), learning_rate);
  )
}

optimizer ato_sgd(double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov) {
  PROTECT(
    auto options = 
      torch::optim::SGDOptions(learning_rate)
      .momentum(momentum)
      .dampening(dampening)
      .weight_decay(weight_decay)
      .nesterov(nesterov);
    return new torch::optim::SGD(vector<torch::Tensor>(), options);
  )
}

void ato_add_parameters(optimizer t, tensor *tensors, int ntensors) {
  PROTECT(
    t->add_parameters(of_carray_tensor(tensors, ntensors));
  )
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.learning_rate_ = learning_rate;
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.learning_rate_ = learning_rate;
    else
     caml_invalid_argument("unexpected optimizer");
  )
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.beta1_ = momentum;
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.momentum_ = momentum;
    else
     caml_invalid_argument("unexpected optimizer");
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

void atc_set_benchmark_cudnn(int b) {
  at::globalContext().setBenchmarkCuDNN(b);
}

#include "torch_api_generated.cpp.h"
