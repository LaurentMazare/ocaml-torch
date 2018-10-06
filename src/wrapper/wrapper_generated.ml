open Ctypes

module C = Torch_bindings.C(Torch_generated)
open C.TensorG

let abs self =
  let t = abs self in
  Gc.finalise C.Tensor.free t;
  t

let abs_ self =
  let t = abs_ self in
  Gc.finalise C.Tensor.free t;
  t

let abs_out result self =
  let t = abs_out result self in
  Gc.finalise C.Tensor.free t;
  t

let acos self =
  let t = acos self in
  Gc.finalise C.Tensor.free t;
  t

let acos_ self =
  let t = acos_ self in
  Gc.finalise C.Tensor.free t;
  t

let acos_out result self =
  let t = acos_out result self in
  Gc.finalise C.Tensor.free t;
  t

let adaptive_avg_pool1d self output_size =
  let t = adaptive_avg_pool1d self (CArray.of_list int output_size |> CArray.start) (List.length output_size) in
  Gc.finalise C.Tensor.free t;
  t

let add self other =
  let t = add self other in
  Gc.finalise C.Tensor.free t;
  t

let add_out result self other =
  let t = add_out result self other in
  Gc.finalise C.Tensor.free t;
  t

let addmm self mat1 mat2 =
  let t = addmm self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let addmm_out result self mat1 mat2 =
  let t = addmm_out result self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let addmv self mat vec =
  let t = addmv self mat vec in
  Gc.finalise C.Tensor.free t;
  t

let addmv_ self mat vec =
  let t = addmv_ self mat vec in
  Gc.finalise C.Tensor.free t;
  t

let addmv_out result self mat vec =
  let t = addmv_out result self mat vec in
  Gc.finalise C.Tensor.free t;
  t

let addr self vec1 vec2 =
  let t = addr self vec1 vec2 in
  Gc.finalise C.Tensor.free t;
  t

let addr_out result self vec1 vec2 =
  let t = addr_out result self vec1 vec2 in
  Gc.finalise C.Tensor.free t;
  t

let all self dim keepdim =
  let t = all self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let all_out result self dim keepdim =
  let t = all_out result self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let alpha_dropout input p train =
  let t = alpha_dropout input p train in
  Gc.finalise C.Tensor.free t;
  t

let alpha_dropout_ self p train =
  let t = alpha_dropout_ self p train in
  Gc.finalise C.Tensor.free t;
  t

let any self dim keepdim =
  let t = any self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let any_out result self dim keepdim =
  let t = any_out result self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let argmax1 self dim keepdim =
  let t = argmax1 self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let argmax2 self =
  let t = argmax2 self in
  Gc.finalise C.Tensor.free t;
  t

let argmin1 self dim keepdim =
  let t = argmin1 self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let argmin2 self =
  let t = argmin2 self in
  Gc.finalise C.Tensor.free t;
  t

let as_strided1 self size stride =
  let t = as_strided1 self (CArray.of_list int size |> CArray.start) (List.length size) (CArray.of_list int stride |> CArray.start) (List.length stride) in
  Gc.finalise C.Tensor.free t;
  t

let as_strided2 self size stride storage_offset =
  let t = as_strided2 self (CArray.of_list int size |> CArray.start) (List.length size) (CArray.of_list int stride |> CArray.start) (List.length stride) storage_offset in
  Gc.finalise C.Tensor.free t;
  t

let as_strided_1 self size stride =
  let t = as_strided_1 self (CArray.of_list int size |> CArray.start) (List.length size) (CArray.of_list int stride |> CArray.start) (List.length stride) in
  Gc.finalise C.Tensor.free t;
  t

let as_strided_2 self size stride storage_offset =
  let t = as_strided_2 self (CArray.of_list int size |> CArray.start) (List.length size) (CArray.of_list int stride |> CArray.start) (List.length stride) storage_offset in
  Gc.finalise C.Tensor.free t;
  t

let asin self =
  let t = asin self in
  Gc.finalise C.Tensor.free t;
  t

let asin_ self =
  let t = asin_ self in
  Gc.finalise C.Tensor.free t;
  t

let asin_out result self =
  let t = asin_out result self in
  Gc.finalise C.Tensor.free t;
  t

let atan self =
  let t = atan self in
  Gc.finalise C.Tensor.free t;
  t

let atan_ self =
  let t = atan_ self in
  Gc.finalise C.Tensor.free t;
  t

let atan_out result self =
  let t = atan_out result self in
  Gc.finalise C.Tensor.free t;
  t

let avg_pool1d self kernel_size stride padding ceil_mode count_include_pad =
  let t = avg_pool1d self (CArray.of_list int kernel_size |> CArray.start) (List.length kernel_size) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) ceil_mode count_include_pad in
  Gc.finalise C.Tensor.free t;
  t

let baddbmm self batch1 batch2 =
  let t = baddbmm self batch1 batch2 in
  Gc.finalise C.Tensor.free t;
  t

let baddbmm_out result self batch1 batch2 =
  let t = baddbmm_out result self batch1 batch2 in
  Gc.finalise C.Tensor.free t;
  t

let bartlett_window1 window_length =
  let t = bartlett_window1 window_length in
  Gc.finalise C.Tensor.free t;
  t

let bartlett_window2 window_length periodic =
  let t = bartlett_window2 window_length periodic in
  Gc.finalise C.Tensor.free t;
  t

let bernoulli1 self =
  let t = bernoulli1 self in
  Gc.finalise C.Tensor.free t;
  t

let bernoulli2 self p =
  let t = bernoulli2 self p in
  Gc.finalise C.Tensor.free t;
  t

let bernoulli_out result self =
  let t = bernoulli_out result self in
  Gc.finalise C.Tensor.free t;
  t

let blackman_window1 window_length =
  let t = blackman_window1 window_length in
  Gc.finalise C.Tensor.free t;
  t

let blackman_window2 window_length periodic =
  let t = blackman_window2 window_length periodic in
  Gc.finalise C.Tensor.free t;
  t

let bmm self mat2 =
  let t = bmm self mat2 in
  Gc.finalise C.Tensor.free t;
  t

let bmm_out result self mat2 =
  let t = bmm_out result self mat2 in
  Gc.finalise C.Tensor.free t;
  t

let ceil self =
  let t = ceil self in
  Gc.finalise C.Tensor.free t;
  t

let ceil_ self =
  let t = ceil_ self in
  Gc.finalise C.Tensor.free t;
  t

let ceil_out result self =
  let t = ceil_out result self in
  Gc.finalise C.Tensor.free t;
  t

let celu self =
  let t = celu self in
  Gc.finalise C.Tensor.free t;
  t

let celu_ self =
  let t = celu_ self in
  Gc.finalise C.Tensor.free t;
  t

let clone self =
  let t = clone self in
  Gc.finalise C.Tensor.free t;
  t

let conv1d input weight bias stride padding dilation groups =
  let t = conv1d input weight bias (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups in
  Gc.finalise C.Tensor.free t;
  t

let conv2d input weight bias stride padding dilation groups =
  let t = conv2d input weight bias (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups in
  Gc.finalise C.Tensor.free t;
  t

let conv3d input weight bias stride padding dilation groups =
  let t = conv3d input weight bias (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups in
  Gc.finalise C.Tensor.free t;
  t

let conv_tbc self weight bias pad =
  let t = conv_tbc self weight bias pad in
  Gc.finalise C.Tensor.free t;
  t

let conv_transpose1d input weight bias stride padding output_padding groups dilation =
  let t = conv_transpose1d input weight bias (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int output_padding |> CArray.start) (List.length output_padding) groups (CArray.of_list int dilation |> CArray.start) (List.length dilation) in
  Gc.finalise C.Tensor.free t;
  t

let conv_transpose2d input weight bias stride padding output_padding groups dilation =
  let t = conv_transpose2d input weight bias (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int output_padding |> CArray.start) (List.length output_padding) groups (CArray.of_list int dilation |> CArray.start) (List.length dilation) in
  Gc.finalise C.Tensor.free t;
  t

let conv_transpose3d input weight bias stride padding output_padding groups dilation =
  let t = conv_transpose3d input weight bias (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int output_padding |> CArray.start) (List.length output_padding) groups (CArray.of_list int dilation |> CArray.start) (List.length dilation) in
  Gc.finalise C.Tensor.free t;
  t

let copy_sparse_to_sparse_ self src non_blocking =
  let t = copy_sparse_to_sparse_ self src non_blocking in
  Gc.finalise C.Tensor.free t;
  t

let cos self =
  let t = cos self in
  Gc.finalise C.Tensor.free t;
  t

let cos_ self =
  let t = cos_ self in
  Gc.finalise C.Tensor.free t;
  t

let cos_out result self =
  let t = cos_out result self in
  Gc.finalise C.Tensor.free t;
  t

let cosh self =
  let t = cosh self in
  Gc.finalise C.Tensor.free t;
  t

let cosh_ self =
  let t = cosh_ self in
  Gc.finalise C.Tensor.free t;
  t

let cosh_out result self =
  let t = cosh_out result self in
  Gc.finalise C.Tensor.free t;
  t

let cosine_embedding_loss input1 input2 target margin reduction =
  let t = cosine_embedding_loss input1 input2 target margin reduction in
  Gc.finalise C.Tensor.free t;
  t

let ctc_loss1 log_probs targets input_lengths target_lengths blank reduction =
  let t = ctc_loss1 log_probs targets (CArray.of_list int input_lengths |> CArray.start) (List.length input_lengths) (CArray.of_list int target_lengths |> CArray.start) (List.length target_lengths) blank reduction in
  Gc.finalise C.Tensor.free t;
  t

let ctc_loss2 log_probs targets input_lengths target_lengths blank reduction =
  let t = ctc_loss2 log_probs targets input_lengths target_lengths blank reduction in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_affine_grid_generator theta n c h w =
  let t = cudnn_affine_grid_generator theta n c h w in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_convolution_backward_bias grad_output =
  let t = cudnn_convolution_backward_bias grad_output in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_convolution_backward_input self_size grad_output weight padding stride dilation groups benchmark deterministic =
  let t = cudnn_convolution_backward_input (CArray.of_list int self_size |> CArray.start) (List.length self_size) grad_output weight (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_convolution_backward_weight weight_size grad_output self padding stride dilation groups benchmark deterministic =
  let t = cudnn_convolution_backward_weight (CArray.of_list int weight_size |> CArray.start) (List.length weight_size) grad_output self (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_convolution_transpose_backward_bias grad_output =
  let t = cudnn_convolution_transpose_backward_bias grad_output in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_convolution_transpose_backward_input grad_output weight padding stride dilation groups benchmark deterministic =
  let t = cudnn_convolution_transpose_backward_input grad_output weight (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let cudnn_convolution_transpose_backward_weight weight_size grad_output self padding stride dilation groups benchmark deterministic =
  let t = cudnn_convolution_transpose_backward_weight (CArray.of_list int weight_size |> CArray.start) (List.length weight_size) grad_output self (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let cumprod self dim =
  let t = cumprod self dim in
  Gc.finalise C.Tensor.free t;
  t

let cumprod_out result self dim =
  let t = cumprod_out result self dim in
  Gc.finalise C.Tensor.free t;
  t

let cumsum self dim =
  let t = cumsum self dim in
  Gc.finalise C.Tensor.free t;
  t

let cumsum_out result self dim =
  let t = cumsum_out result self dim in
  Gc.finalise C.Tensor.free t;
  t

let det self =
  let t = det self in
  Gc.finalise C.Tensor.free t;
  t

let detach self =
  let t = detach self in
  Gc.finalise C.Tensor.free t;
  t

let detach_ self =
  let t = detach_ self in
  Gc.finalise C.Tensor.free t;
  t

let diagflat self offset =
  let t = diagflat self offset in
  Gc.finalise C.Tensor.free t;
  t

let diagonal self offset dim1 dim2 =
  let t = diagonal self offset dim1 dim2 in
  Gc.finalise C.Tensor.free t;
  t

let div self other =
  let t = div self other in
  Gc.finalise C.Tensor.free t;
  t

let div_out result self other =
  let t = div_out result self other in
  Gc.finalise C.Tensor.free t;
  t

let dot self tensor =
  let t = dot self tensor in
  Gc.finalise C.Tensor.free t;
  t

let dot_out result self tensor =
  let t = dot_out result self tensor in
  Gc.finalise C.Tensor.free t;
  t

let dropout input p train =
  let t = dropout input p train in
  Gc.finalise C.Tensor.free t;
  t

let dropout_ self p train =
  let t = dropout_ self p train in
  Gc.finalise C.Tensor.free t;
  t

let empty size =
  let t = empty (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let empty_like self =
  let t = empty_like self in
  Gc.finalise C.Tensor.free t;
  t

let empty_out result size =
  let t = empty_out result (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let empty_strided size stride =
  let t = empty_strided (CArray.of_list int size |> CArray.start) (List.length size) (CArray.of_list int stride |> CArray.start) (List.length stride) in
  Gc.finalise C.Tensor.free t;
  t

let erf self =
  let t = erf self in
  Gc.finalise C.Tensor.free t;
  t

let erf_ self =
  let t = erf_ self in
  Gc.finalise C.Tensor.free t;
  t

let erf_out result self =
  let t = erf_out result self in
  Gc.finalise C.Tensor.free t;
  t

let erfc self =
  let t = erfc self in
  Gc.finalise C.Tensor.free t;
  t

let erfc_ self =
  let t = erfc_ self in
  Gc.finalise C.Tensor.free t;
  t

let erfc_out result self =
  let t = erfc_out result self in
  Gc.finalise C.Tensor.free t;
  t

let exp self =
  let t = exp self in
  Gc.finalise C.Tensor.free t;
  t

let exp_ self =
  let t = exp_ self in
  Gc.finalise C.Tensor.free t;
  t

let exp_out result self =
  let t = exp_out result self in
  Gc.finalise C.Tensor.free t;
  t

let expm1 self =
  let t = expm1 self in
  Gc.finalise C.Tensor.free t;
  t

let expm1_ self =
  let t = expm1_ self in
  Gc.finalise C.Tensor.free t;
  t

let expm1_out result self =
  let t = expm1_out result self in
  Gc.finalise C.Tensor.free t;
  t

let eye1 n =
  let t = eye1 n in
  Gc.finalise C.Tensor.free t;
  t

let eye2 n m =
  let t = eye2 n m in
  Gc.finalise C.Tensor.free t;
  t

let eye_out1 result n =
  let t = eye_out1 result n in
  Gc.finalise C.Tensor.free t;
  t

let eye_out2 result n m =
  let t = eye_out2 result n m in
  Gc.finalise C.Tensor.free t;
  t

let feature_alpha_dropout input p train =
  let t = feature_alpha_dropout input p train in
  Gc.finalise C.Tensor.free t;
  t

let feature_alpha_dropout_ self p train =
  let t = feature_alpha_dropout_ self p train in
  Gc.finalise C.Tensor.free t;
  t

let feature_dropout input p train =
  let t = feature_dropout input p train in
  Gc.finalise C.Tensor.free t;
  t

let feature_dropout_ self p train =
  let t = feature_dropout_ self p train in
  Gc.finalise C.Tensor.free t;
  t

let fft self signal_ndim normalized =
  let t = fft self signal_ndim normalized in
  Gc.finalise C.Tensor.free t;
  t

let fill_ self value =
  let t = fill_ self value in
  Gc.finalise C.Tensor.free t;
  t

let flatten self start_dim end_dim =
  let t = flatten self start_dim end_dim in
  Gc.finalise C.Tensor.free t;
  t

let flip self dims =
  let t = flip self (CArray.of_list int dims |> CArray.start) (List.length dims) in
  Gc.finalise C.Tensor.free t;
  t

let floor self =
  let t = floor self in
  Gc.finalise C.Tensor.free t;
  t

let floor_ self =
  let t = floor_ self in
  Gc.finalise C.Tensor.free t;
  t

let floor_out result self =
  let t = floor_out result self in
  Gc.finalise C.Tensor.free t;
  t

let frobenius_norm1 self =
  let t = frobenius_norm1 self in
  Gc.finalise C.Tensor.free t;
  t

let frobenius_norm2 self dim keepdim =
  let t = frobenius_norm2 self (CArray.of_list int dim |> CArray.start) (List.length dim) keepdim in
  Gc.finalise C.Tensor.free t;
  t

let frobenius_norm_out result self dim keepdim =
  let t = frobenius_norm_out result self (CArray.of_list int dim |> CArray.start) (List.length dim) keepdim in
  Gc.finalise C.Tensor.free t;
  t

let ger self vec2 =
  let t = ger self vec2 in
  Gc.finalise C.Tensor.free t;
  t

let ger_out result self vec2 =
  let t = ger_out result self vec2 in
  Gc.finalise C.Tensor.free t;
  t

let grid_sampler input grid interpolation_mode padding_mode =
  let t = grid_sampler input grid interpolation_mode padding_mode in
  Gc.finalise C.Tensor.free t;
  t

let grid_sampler_2d input grid interpolation_mode padding_mode =
  let t = grid_sampler_2d input grid interpolation_mode padding_mode in
  Gc.finalise C.Tensor.free t;
  t

let grid_sampler_3d input grid interpolation_mode padding_mode =
  let t = grid_sampler_3d input grid interpolation_mode padding_mode in
  Gc.finalise C.Tensor.free t;
  t

let gru_cell input hx w_ih w_hh =
  let t = gru_cell input hx w_ih w_hh in
  Gc.finalise C.Tensor.free t;
  t

let hamming_window1 window_length =
  let t = hamming_window1 window_length in
  Gc.finalise C.Tensor.free t;
  t

let hamming_window2 window_length periodic =
  let t = hamming_window2 window_length periodic in
  Gc.finalise C.Tensor.free t;
  t

let hamming_window3 window_length periodic alpha =
  let t = hamming_window3 window_length periodic alpha in
  Gc.finalise C.Tensor.free t;
  t

let hamming_window4 window_length periodic alpha beta =
  let t = hamming_window4 window_length periodic alpha beta in
  Gc.finalise C.Tensor.free t;
  t

let hann_window1 window_length =
  let t = hann_window1 window_length in
  Gc.finalise C.Tensor.free t;
  t

let hann_window2 window_length periodic =
  let t = hann_window2 window_length periodic in
  Gc.finalise C.Tensor.free t;
  t

let hardshrink self =
  let t = hardshrink self in
  Gc.finalise C.Tensor.free t;
  t

let hinge_embedding_loss self target margin reduction =
  let t = hinge_embedding_loss self target margin reduction in
  Gc.finalise C.Tensor.free t;
  t

let hspmm mat1 mat2 =
  let t = hspmm mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let hspmm_out result mat1 mat2 =
  let t = hspmm_out result mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let ifft self signal_ndim normalized =
  let t = ifft self signal_ndim normalized in
  Gc.finalise C.Tensor.free t;
  t

let inverse self =
  let t = inverse self in
  Gc.finalise C.Tensor.free t;
  t

let inverse_out result self =
  let t = inverse_out result self in
  Gc.finalise C.Tensor.free t;
  t

let irfft self signal_ndim normalized onesided signal_sizes =
  let t = irfft self signal_ndim normalized onesided (CArray.of_list int signal_sizes |> CArray.start) (List.length signal_sizes) in
  Gc.finalise C.Tensor.free t;
  t

let isclose self other rtol atol equal_nan =
  let t = isclose self other rtol atol equal_nan in
  Gc.finalise C.Tensor.free t;
  t

let kl_div self target reduction =
  let t = kl_div self target reduction in
  Gc.finalise C.Tensor.free t;
  t

let kl_div_backward grad_output self target reduction =
  let t = kl_div_backward grad_output self target reduction in
  Gc.finalise C.Tensor.free t;
  t

let linear input weight bias =
  let t = linear input weight bias in
  Gc.finalise C.Tensor.free t;
  t

let log self =
  let t = log self in
  Gc.finalise C.Tensor.free t;
  t

let log10 self =
  let t = log10 self in
  Gc.finalise C.Tensor.free t;
  t

let log10_ self =
  let t = log10_ self in
  Gc.finalise C.Tensor.free t;
  t

let log10_out result self =
  let t = log10_out result self in
  Gc.finalise C.Tensor.free t;
  t

let log1p self =
  let t = log1p self in
  Gc.finalise C.Tensor.free t;
  t

let log1p_ self =
  let t = log1p_ self in
  Gc.finalise C.Tensor.free t;
  t

let log1p_out result self =
  let t = log1p_out result self in
  Gc.finalise C.Tensor.free t;
  t

let log2 self =
  let t = log2 self in
  Gc.finalise C.Tensor.free t;
  t

let log2_ self =
  let t = log2_ self in
  Gc.finalise C.Tensor.free t;
  t

let log2_out result self =
  let t = log2_out result self in
  Gc.finalise C.Tensor.free t;
  t

let log_ self =
  let t = log_ self in
  Gc.finalise C.Tensor.free t;
  t

let log_out result self =
  let t = log_out result self in
  Gc.finalise C.Tensor.free t;
  t

let log_softmax self dim =
  let t = log_softmax self dim in
  Gc.finalise C.Tensor.free t;
  t

let log_softmax_backward_data grad_output output dim self =
  let t = log_softmax_backward_data grad_output output dim self in
  Gc.finalise C.Tensor.free t;
  t

let logdet self =
  let t = logdet self in
  Gc.finalise C.Tensor.free t;
  t

let logsumexp self dim keepdim =
  let t = logsumexp self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let logsumexp_out result self dim keepdim =
  let t = logsumexp_out result self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let margin_ranking_loss input1 input2 target margin reduction =
  let t = margin_ranking_loss input1 input2 target margin reduction in
  Gc.finalise C.Tensor.free t;
  t

let matmul self other =
  let t = matmul self other in
  Gc.finalise C.Tensor.free t;
  t

let matmul_out result self other =
  let t = matmul_out result self other in
  Gc.finalise C.Tensor.free t;
  t

let matrix_power self n =
  let t = matrix_power self n in
  Gc.finalise C.Tensor.free t;
  t

let matrix_rank1 self tol symmetric =
  let t = matrix_rank1 self tol symmetric in
  Gc.finalise C.Tensor.free t;
  t

let matrix_rank2 self symmetric =
  let t = matrix_rank2 self symmetric in
  Gc.finalise C.Tensor.free t;
  t

let max_pool1d self kernel_size stride padding dilation ceil_mode =
  let t = max_pool1d self (CArray.of_list int kernel_size |> CArray.start) (List.length kernel_size) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int dilation |> CArray.start) (List.length dilation) ceil_mode in
  Gc.finalise C.Tensor.free t;
  t

let max_pool2d self kernel_size stride padding dilation ceil_mode =
  let t = max_pool2d self (CArray.of_list int kernel_size |> CArray.start) (List.length kernel_size) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int dilation |> CArray.start) (List.length dilation) ceil_mode in
  Gc.finalise C.Tensor.free t;
  t

let max_pool3d self kernel_size stride padding dilation ceil_mode =
  let t = max_pool3d self (CArray.of_list int kernel_size |> CArray.start) (List.length kernel_size) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int dilation |> CArray.start) (List.length dilation) ceil_mode in
  Gc.finalise C.Tensor.free t;
  t

let max_values self dim keepdim =
  let t = max_values self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let mean1 self =
  let t = mean1 self in
  Gc.finalise C.Tensor.free t;
  t

let mean2 self dim keepdim =
  let t = mean2 self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let mean_out result self dim keepdim =
  let t = mean_out result self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let min_values self dim keepdim =
  let t = min_values self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let miopen_convolution_backward_bias grad_output =
  let t = miopen_convolution_backward_bias grad_output in
  Gc.finalise C.Tensor.free t;
  t

let miopen_convolution_backward_input self_size grad_output weight padding stride dilation groups benchmark deterministic =
  let t = miopen_convolution_backward_input (CArray.of_list int self_size |> CArray.start) (List.length self_size) grad_output weight (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let miopen_convolution_backward_weight weight_size grad_output self padding stride dilation groups benchmark deterministic =
  let t = miopen_convolution_backward_weight (CArray.of_list int weight_size |> CArray.start) (List.length weight_size) grad_output self (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let miopen_convolution_transpose_backward_input grad_output weight padding stride dilation groups benchmark deterministic =
  let t = miopen_convolution_transpose_backward_input grad_output weight (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let miopen_convolution_transpose_backward_weight weight_size grad_output self padding stride dilation groups benchmark deterministic =
  let t = miopen_convolution_transpose_backward_weight (CArray.of_list int weight_size |> CArray.start) (List.length weight_size) grad_output self (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups benchmark deterministic in
  Gc.finalise C.Tensor.free t;
  t

let mkldnn_convolution_backward_input self_size grad_output weight padding stride dilation groups bias_defined =
  let t = mkldnn_convolution_backward_input (CArray.of_list int self_size |> CArray.start) (List.length self_size) grad_output weight (CArray.of_list int padding |> CArray.start) (List.length padding) (CArray.of_list int stride |> CArray.start) (List.length stride) (CArray.of_list int dilation |> CArray.start) (List.length dilation) groups bias_defined in
  Gc.finalise C.Tensor.free t;
  t

let mm self mat2 =
  let t = mm self mat2 in
  Gc.finalise C.Tensor.free t;
  t

let mm_out result self mat2 =
  let t = mm_out result self mat2 in
  Gc.finalise C.Tensor.free t;
  t

let mul self other =
  let t = mul self other in
  Gc.finalise C.Tensor.free t;
  t

let mul_out result self other =
  let t = mul_out result self other in
  Gc.finalise C.Tensor.free t;
  t

let mv self vec =
  let t = mv self vec in
  Gc.finalise C.Tensor.free t;
  t

let mv_out result self vec =
  let t = mv_out result self vec in
  Gc.finalise C.Tensor.free t;
  t

let mvlgamma self p =
  let t = mvlgamma self p in
  Gc.finalise C.Tensor.free t;
  t

let narrow self dim start length =
  let t = narrow self dim start length in
  Gc.finalise C.Tensor.free t;
  t

let native_clone self =
  let t = native_clone self in
  Gc.finalise C.Tensor.free t;
  t

let native_norm self =
  let t = native_norm self in
  Gc.finalise C.Tensor.free t;
  t

let native_resize_as_ self the_template =
  let t = native_resize_as_ self the_template in
  Gc.finalise C.Tensor.free t;
  t

let native_zero_ self =
  let t = native_zero_ self in
  Gc.finalise C.Tensor.free t;
  t

let norm self =
  let t = norm self in
  Gc.finalise C.Tensor.free t;
  t

let norm_except_dim v pow dim =
  let t = norm_except_dim v pow dim in
  Gc.finalise C.Tensor.free t;
  t

let nuclear_norm self keepdim =
  let t = nuclear_norm self keepdim in
  Gc.finalise C.Tensor.free t;
  t

let nuclear_norm_out result self keepdim =
  let t = nuclear_norm_out result self keepdim in
  Gc.finalise C.Tensor.free t;
  t

let ones size =
  let t = ones (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let ones_like self =
  let t = ones_like self in
  Gc.finalise C.Tensor.free t;
  t

let ones_out result size =
  let t = ones_out result (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let pairwise_distance x1 x2 p eps keepdim =
  let t = pairwise_distance x1 x2 p eps keepdim in
  Gc.finalise C.Tensor.free t;
  t

let pdist self p =
  let t = pdist self p in
  Gc.finalise C.Tensor.free t;
  t

let pin_memory self =
  let t = pin_memory self in
  Gc.finalise C.Tensor.free t;
  t

let pinverse self rcond =
  let t = pinverse self rcond in
  Gc.finalise C.Tensor.free t;
  t

let pixel_shuffle self upscale_factor =
  let t = pixel_shuffle self upscale_factor in
  Gc.finalise C.Tensor.free t;
  t

let poisson self =
  let t = poisson self in
  Gc.finalise C.Tensor.free t;
  t

let prelu self weight =
  let t = prelu self weight in
  Gc.finalise C.Tensor.free t;
  t

let prod1 self =
  let t = prod1 self in
  Gc.finalise C.Tensor.free t;
  t

let prod2 self dim keepdim =
  let t = prod2 self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let prod_out result self dim keepdim =
  let t = prod_out result self dim keepdim in
  Gc.finalise C.Tensor.free t;
  t

let rand size =
  let t = rand (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let rand_like self =
  let t = rand_like self in
  Gc.finalise C.Tensor.free t;
  t

let rand_out result size =
  let t = rand_out result (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randint1 high size =
  let t = randint1 high (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randint2 low high size =
  let t = randint2 low high (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randint_like1 self high =
  let t = randint_like1 self high in
  Gc.finalise C.Tensor.free t;
  t

let randint_like2 self low high =
  let t = randint_like2 self low high in
  Gc.finalise C.Tensor.free t;
  t

let randint_out1 result high size =
  let t = randint_out1 result high (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randint_out2 result low high size =
  let t = randint_out2 result low high (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randn size =
  let t = randn (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randn_like self =
  let t = randn_like self in
  Gc.finalise C.Tensor.free t;
  t

let randn_out result size =
  let t = randn_out result (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let randperm n =
  let t = randperm n in
  Gc.finalise C.Tensor.free t;
  t

let randperm_out result n =
  let t = randperm_out result n in
  Gc.finalise C.Tensor.free t;
  t

let relu self =
  let t = relu self in
  Gc.finalise C.Tensor.free t;
  t

let relu_ self =
  let t = relu_ self in
  Gc.finalise C.Tensor.free t;
  t

let reshape self shape =
  let t = reshape self (CArray.of_list int shape |> CArray.start) (List.length shape) in
  Gc.finalise C.Tensor.free t;
  t

let resize_as_ self the_template =
  let t = resize_as_ self the_template in
  Gc.finalise C.Tensor.free t;
  t

let rfft self signal_ndim normalized onesided =
  let t = rfft self signal_ndim normalized onesided in
  Gc.finalise C.Tensor.free t;
  t

let rnn_relu_cell input hx w_ih w_hh =
  let t = rnn_relu_cell input hx w_ih w_hh in
  Gc.finalise C.Tensor.free t;
  t

let rnn_tanh_cell input hx w_ih w_hh =
  let t = rnn_tanh_cell input hx w_ih w_hh in
  Gc.finalise C.Tensor.free t;
  t

let roipooling2d_backward input rois pooledheight pooledwidth spatialscale gradoutput argmaxes =
  let t = roipooling2d_backward input rois pooledheight pooledwidth spatialscale gradoutput argmaxes in
  Gc.finalise C.Tensor.free t;
  t

let round self =
  let t = round self in
  Gc.finalise C.Tensor.free t;
  t

let round_ self =
  let t = round_ self in
  Gc.finalise C.Tensor.free t;
  t

let round_out result self =
  let t = round_out result self in
  Gc.finalise C.Tensor.free t;
  t

let rrelu self training =
  let t = rrelu self training in
  Gc.finalise C.Tensor.free t;
  t

let rrelu_ self training =
  let t = rrelu_ self training in
  Gc.finalise C.Tensor.free t;
  t

let rsqrt self =
  let t = rsqrt self in
  Gc.finalise C.Tensor.free t;
  t

let rsqrt_ self =
  let t = rsqrt_ self in
  Gc.finalise C.Tensor.free t;
  t

let rsqrt_out result self =
  let t = rsqrt_out result self in
  Gc.finalise C.Tensor.free t;
  t

let s_native_addmm self mat1 mat2 =
  let t = s_native_addmm self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let s_native_addmm_ self mat1 mat2 =
  let t = s_native_addmm_ self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let s_native_addmm_out result self mat1 mat2 =
  let t = s_native_addmm_out result self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let select self dim index =
  let t = select self dim index in
  Gc.finalise C.Tensor.free t;
  t

let selu self =
  let t = selu self in
  Gc.finalise C.Tensor.free t;
  t

let selu_ self =
  let t = selu_ self in
  Gc.finalise C.Tensor.free t;
  t

let sigmoid self =
  let t = sigmoid self in
  Gc.finalise C.Tensor.free t;
  t

let sigmoid_ self =
  let t = sigmoid_ self in
  Gc.finalise C.Tensor.free t;
  t

let sigmoid_out result self =
  let t = sigmoid_out result self in
  Gc.finalise C.Tensor.free t;
  t

let sin self =
  let t = sin self in
  Gc.finalise C.Tensor.free t;
  t

let sin_ self =
  let t = sin_ self in
  Gc.finalise C.Tensor.free t;
  t

let sin_out result self =
  let t = sin_out result self in
  Gc.finalise C.Tensor.free t;
  t

let sinh self =
  let t = sinh self in
  Gc.finalise C.Tensor.free t;
  t

let sinh_ self =
  let t = sinh_ self in
  Gc.finalise C.Tensor.free t;
  t

let sinh_out result self =
  let t = sinh_out result self in
  Gc.finalise C.Tensor.free t;
  t

let slice self dim start end_ step =
  let t = slice self dim start end_ step in
  Gc.finalise C.Tensor.free t;
  t

let smm self mat2 =
  let t = smm self mat2 in
  Gc.finalise C.Tensor.free t;
  t

let softmax self dim =
  let t = softmax self dim in
  Gc.finalise C.Tensor.free t;
  t

let softmax_backward_data grad_output output dim self =
  let t = softmax_backward_data grad_output output dim self in
  Gc.finalise C.Tensor.free t;
  t

let sqrt self =
  let t = sqrt self in
  Gc.finalise C.Tensor.free t;
  t

let sqrt_ self =
  let t = sqrt_ self in
  Gc.finalise C.Tensor.free t;
  t

let sqrt_out result self =
  let t = sqrt_out result self in
  Gc.finalise C.Tensor.free t;
  t

let squeeze1 self =
  let t = squeeze1 self in
  Gc.finalise C.Tensor.free t;
  t

let squeeze2 self dim =
  let t = squeeze2 self dim in
  Gc.finalise C.Tensor.free t;
  t

let sspaddmm self mat1 mat2 =
  let t = sspaddmm self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let sspaddmm_out result self mat1 mat2 =
  let t = sspaddmm_out result self mat1 mat2 in
  Gc.finalise C.Tensor.free t;
  t

let std1 self unbiased =
  let t = std1 self unbiased in
  Gc.finalise C.Tensor.free t;
  t

let std2 self dim unbiased keepdim =
  let t = std2 self dim unbiased keepdim in
  Gc.finalise C.Tensor.free t;
  t

let std_out result self dim unbiased keepdim =
  let t = std_out result self dim unbiased keepdim in
  Gc.finalise C.Tensor.free t;
  t

let sub self other =
  let t = sub self other in
  Gc.finalise C.Tensor.free t;
  t

let sub_out result self other =
  let t = sub_out result self other in
  Gc.finalise C.Tensor.free t;
  t

let sum1 self =
  let t = sum1 self in
  Gc.finalise C.Tensor.free t;
  t

let sum2 self dim keepdim =
  let t = sum2 self (CArray.of_list int dim |> CArray.start) (List.length dim) keepdim in
  Gc.finalise C.Tensor.free t;
  t

let sum_out result self dim keepdim =
  let t = sum_out result self (CArray.of_list int dim |> CArray.start) (List.length dim) keepdim in
  Gc.finalise C.Tensor.free t;
  t

let tan self =
  let t = tan self in
  Gc.finalise C.Tensor.free t;
  t

let tan_ self =
  let t = tan_ self in
  Gc.finalise C.Tensor.free t;
  t

let tan_out result self =
  let t = tan_out result self in
  Gc.finalise C.Tensor.free t;
  t

let tanh self =
  let t = tanh self in
  Gc.finalise C.Tensor.free t;
  t

let tanh_ self =
  let t = tanh_ self in
  Gc.finalise C.Tensor.free t;
  t

let tanh_out result self =
  let t = tanh_out result self in
  Gc.finalise C.Tensor.free t;
  t

let tensordot self other dims_self dims_other =
  let t = tensordot self other (CArray.of_list int dims_self |> CArray.start) (List.length dims_self) (CArray.of_list int dims_other |> CArray.start) (List.length dims_other) in
  Gc.finalise C.Tensor.free t;
  t

let transpose self dim0 dim1 =
  let t = transpose self dim0 dim1 in
  Gc.finalise C.Tensor.free t;
  t

let triplet_margin_loss anchor positive negative margin p eps swap reduction =
  let t = triplet_margin_loss anchor positive negative margin p eps swap reduction in
  Gc.finalise C.Tensor.free t;
  t

let trunc self =
  let t = trunc self in
  Gc.finalise C.Tensor.free t;
  t

let trunc_ self =
  let t = trunc_ self in
  Gc.finalise C.Tensor.free t;
  t

let trunc_out result self =
  let t = trunc_out result self in
  Gc.finalise C.Tensor.free t;
  t

let unsqueeze self dim =
  let t = unsqueeze self dim in
  Gc.finalise C.Tensor.free t;
  t

let var1 self unbiased =
  let t = var1 self unbiased in
  Gc.finalise C.Tensor.free t;
  t

let var2 self dim unbiased keepdim =
  let t = var2 self dim unbiased keepdim in
  Gc.finalise C.Tensor.free t;
  t

let var_out result self dim unbiased keepdim =
  let t = var_out result self dim unbiased keepdim in
  Gc.finalise C.Tensor.free t;
  t

let zero_ self =
  let t = zero_ self in
  Gc.finalise C.Tensor.free t;
  t

let zeros size =
  let t = zeros (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

let zeros_like self =
  let t = zeros_like self in
  Gc.finalise C.Tensor.free t;
  t

let zeros_out result size =
  let t = zeros_out result (CArray.of_list int size |> CArray.start) (List.length size) in
  Gc.finalise C.Tensor.free t;
  t

