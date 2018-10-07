open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  type t = unit ptr
  let t : t typ = ptr void
  let abs =
    foreign "atg_abs"
    (t @-> returning t)

  let abs_ =
    foreign "atg_abs_"
    (t @-> returning t)

  let abs_out =
    foreign "atg_abs_out"
    (t @-> t @-> returning t)

  let acos =
    foreign "atg_acos"
    (t @-> returning t)

  let acos_ =
    foreign "atg_acos_"
    (t @-> returning t)

  let acos_out =
    foreign "atg_acos_out"
    (t @-> t @-> returning t)

  let adaptive_avg_pool1d =
    foreign "atg_adaptive_avg_pool1d"
    (t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool2d =
    foreign "atg_adaptive_avg_pool2d"
    (t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool2d_backward =
    foreign "atg_adaptive_avg_pool2d_backward"
    (t @-> t @-> returning t)

  let adaptive_avg_pool2d_backward_out =
    foreign "atg_adaptive_avg_pool2d_backward_out"
    (t @-> t @-> t @-> returning t)

  let adaptive_avg_pool2d_forward =
    foreign "atg_adaptive_avg_pool2d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool2d_forward_out =
    foreign "atg_adaptive_avg_pool2d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool2d_out =
    foreign "atg_adaptive_avg_pool2d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool3d =
    foreign "atg_adaptive_avg_pool3d"
    (t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool3d_backward =
    foreign "atg_adaptive_avg_pool3d_backward"
    (t @-> t @-> returning t)

  let adaptive_avg_pool3d_backward_out =
    foreign "atg_adaptive_avg_pool3d_backward_out"
    (t @-> t @-> t @-> returning t)

  let adaptive_avg_pool3d_forward =
    foreign "atg_adaptive_avg_pool3d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool3d_forward_out =
    foreign "atg_adaptive_avg_pool3d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let adaptive_avg_pool3d_out =
    foreign "atg_adaptive_avg_pool3d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let add =
    foreign "atg_add"
    (t @-> t @-> returning t)

  let add_ =
    foreign "atg_add_"
    (t @-> t @-> returning t)

  let add_out =
    foreign "atg_add_out"
    (t @-> t @-> t @-> returning t)

  let addbmm =
    foreign "atg_addbmm"
    (t @-> t @-> t @-> returning t)

  let addbmm_ =
    foreign "atg_addbmm_"
    (t @-> t @-> t @-> returning t)

  let addbmm_out =
    foreign "atg_addbmm_out"
    (t @-> t @-> t @-> t @-> returning t)

  let addcdiv =
    foreign "atg_addcdiv"
    (t @-> t @-> t @-> returning t)

  let addcdiv_ =
    foreign "atg_addcdiv_"
    (t @-> t @-> t @-> returning t)

  let addcdiv_out =
    foreign "atg_addcdiv_out"
    (t @-> t @-> t @-> t @-> returning t)

  let addcmul =
    foreign "atg_addcmul"
    (t @-> t @-> t @-> returning t)

  let addcmul_ =
    foreign "atg_addcmul_"
    (t @-> t @-> t @-> returning t)

  let addcmul_out =
    foreign "atg_addcmul_out"
    (t @-> t @-> t @-> t @-> returning t)

  let addmm =
    foreign "atg_addmm"
    (t @-> t @-> t @-> returning t)

  let addmm_ =
    foreign "atg_addmm_"
    (t @-> t @-> t @-> returning t)

  let addmm_out =
    foreign "atg_addmm_out"
    (t @-> t @-> t @-> t @-> returning t)

  let addmv =
    foreign "atg_addmv"
    (t @-> t @-> t @-> returning t)

  let addmv_ =
    foreign "atg_addmv_"
    (t @-> t @-> t @-> returning t)

  let addmv_out =
    foreign "atg_addmv_out"
    (t @-> t @-> t @-> t @-> returning t)

  let addr =
    foreign "atg_addr"
    (t @-> t @-> t @-> returning t)

  let addr_ =
    foreign "atg_addr_"
    (t @-> t @-> t @-> returning t)

  let addr_out =
    foreign "atg_addr_out"
    (t @-> t @-> t @-> t @-> returning t)

  let alias =
    foreign "atg_alias"
    (t @-> returning t)

  let all =
    foreign "atg_all"
    (t @-> int64_t @-> int @-> returning t)

  let all_out =
    foreign "atg_all_out"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let alpha_dropout =
    foreign "atg_alpha_dropout"
    (t @-> double @-> int @-> returning t)

  let alpha_dropout_ =
    foreign "atg_alpha_dropout_"
    (t @-> double @-> int @-> returning t)

  let any =
    foreign "atg_any"
    (t @-> int64_t @-> int @-> returning t)

  let any_out =
    foreign "atg_any_out"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let argmax1 =
    foreign "atg_argmax1"
    (t @-> int64_t @-> int @-> returning t)

  let argmax2 =
    foreign "atg_argmax2"
    (t @-> returning t)

  let argmin1 =
    foreign "atg_argmin1"
    (t @-> int64_t @-> int @-> returning t)

  let argmin2 =
    foreign "atg_argmin2"
    (t @-> returning t)

  let as_strided1 =
    foreign "atg_as_strided1"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let as_strided2 =
    foreign "atg_as_strided2"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let as_strided_1 =
    foreign "atg_as_strided_1"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let as_strided_2 =
    foreign "atg_as_strided_2"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let asin =
    foreign "atg_asin"
    (t @-> returning t)

  let asin_ =
    foreign "atg_asin_"
    (t @-> returning t)

  let asin_out =
    foreign "atg_asin_out"
    (t @-> t @-> returning t)

  let atan =
    foreign "atg_atan"
    (t @-> returning t)

  let atan2 =
    foreign "atg_atan2"
    (t @-> t @-> returning t)

  let atan2_ =
    foreign "atg_atan2_"
    (t @-> t @-> returning t)

  let atan2_out =
    foreign "atg_atan2_out"
    (t @-> t @-> t @-> returning t)

  let atan_ =
    foreign "atg_atan_"
    (t @-> returning t)

  let atan_out =
    foreign "atg_atan_out"
    (t @-> t @-> returning t)

  let avg_pool1d =
    foreign "atg_avg_pool1d"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool2d =
    foreign "atg_avg_pool2d"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool2d_backward =
    foreign "atg_avg_pool2d_backward"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool2d_backward_out =
    foreign "atg_avg_pool2d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool2d_forward =
    foreign "atg_avg_pool2d_forward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool2d_forward_out =
    foreign "atg_avg_pool2d_forward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool2d_out =
    foreign "atg_avg_pool2d_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool3d =
    foreign "atg_avg_pool3d"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool3d_backward =
    foreign "atg_avg_pool3d_backward"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool3d_backward_out =
    foreign "atg_avg_pool3d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool3d_forward =
    foreign "atg_avg_pool3d_forward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool3d_forward_out =
    foreign "atg_avg_pool3d_forward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let avg_pool3d_out =
    foreign "atg_avg_pool3d_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> int @-> returning t)

  let baddbmm =
    foreign "atg_baddbmm"
    (t @-> t @-> t @-> returning t)

  let baddbmm_ =
    foreign "atg_baddbmm_"
    (t @-> t @-> t @-> returning t)

  let baddbmm_out =
    foreign "atg_baddbmm_out"
    (t @-> t @-> t @-> t @-> returning t)

  let bartlett_window1 =
    foreign "atg_bartlett_window1"
    (int64_t @-> int @-> returning t)

  let bartlett_window2 =
    foreign "atg_bartlett_window2"
    (int64_t @-> int @-> int @-> returning t)

  let batch_norm =
    foreign "atg_batch_norm"
    (t @-> t @-> t @-> t @-> t @-> int @-> double @-> double @-> int @-> returning t)

  let bernoulli1 =
    foreign "atg_bernoulli1"
    (t @-> returning t)

  let bernoulli2 =
    foreign "atg_bernoulli2"
    (t @-> double @-> returning t)

  let bernoulli_1 =
    foreign "atg_bernoulli_1"
    (t @-> t @-> returning t)

  let bernoulli_2 =
    foreign "atg_bernoulli_2"
    (t @-> double @-> returning t)

  let bernoulli_out =
    foreign "atg_bernoulli_out"
    (t @-> t @-> returning t)

  let bilinear =
    foreign "atg_bilinear"
    (t @-> t @-> t @-> t @-> returning t)

  let binary_cross_entropy =
    foreign "atg_binary_cross_entropy"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let binary_cross_entropy_backward =
    foreign "atg_binary_cross_entropy_backward"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let binary_cross_entropy_backward_out =
    foreign "atg_binary_cross_entropy_backward_out"
    (t @-> t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let binary_cross_entropy_forward =
    foreign "atg_binary_cross_entropy_forward"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let binary_cross_entropy_forward_out =
    foreign "atg_binary_cross_entropy_forward_out"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let binary_cross_entropy_out =
    foreign "atg_binary_cross_entropy_out"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let blackman_window1 =
    foreign "atg_blackman_window1"
    (int64_t @-> int @-> returning t)

  let blackman_window2 =
    foreign "atg_blackman_window2"
    (int64_t @-> int @-> int @-> returning t)

  let bmm =
    foreign "atg_bmm"
    (t @-> t @-> returning t)

  let bmm_out =
    foreign "atg_bmm_out"
    (t @-> t @-> t @-> returning t)

  let cauchy_ =
    foreign "atg_cauchy_"
    (t @-> double @-> double @-> returning t)

  let ceil =
    foreign "atg_ceil"
    (t @-> returning t)

  let ceil_ =
    foreign "atg_ceil_"
    (t @-> returning t)

  let ceil_out =
    foreign "atg_ceil_out"
    (t @-> t @-> returning t)

  let celu =
    foreign "atg_celu"
    (t @-> returning t)

  let celu_ =
    foreign "atg_celu_"
    (t @-> returning t)

  let clone =
    foreign "atg_clone"
    (t @-> returning t)

  let coalesce =
    foreign "atg_coalesce"
    (t @-> returning t)

  let contiguous =
    foreign "atg_contiguous"
    (t @-> returning t)

  let conv1d =
    foreign "atg_conv1d"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let conv2d =
    foreign "atg_conv2d"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let conv3d =
    foreign "atg_conv3d"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let conv_tbc =
    foreign "atg_conv_tbc"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let conv_transpose1d =
    foreign "atg_conv_transpose1d"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> ptr int @-> int @-> returning t)

  let conv_transpose2d =
    foreign "atg_conv_transpose2d"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> ptr int @-> int @-> returning t)

  let conv_transpose3d =
    foreign "atg_conv_transpose3d"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> ptr int @-> int @-> returning t)

  let convolution =
    foreign "atg_convolution"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let copy_sparse_to_sparse_ =
    foreign "atg_copy_sparse_to_sparse_"
    (t @-> t @-> int @-> returning t)

  let cos =
    foreign "atg_cos"
    (t @-> returning t)

  let cos_ =
    foreign "atg_cos_"
    (t @-> returning t)

  let cos_out =
    foreign "atg_cos_out"
    (t @-> t @-> returning t)

  let cosh =
    foreign "atg_cosh"
    (t @-> returning t)

  let cosh_ =
    foreign "atg_cosh_"
    (t @-> returning t)

  let cosh_out =
    foreign "atg_cosh_out"
    (t @-> t @-> returning t)

  let cosine_embedding_loss =
    foreign "atg_cosine_embedding_loss"
    (t @-> t @-> t @-> double @-> int64_t @-> returning t)

  let cross =
    foreign "atg_cross"
    (t @-> t @-> int64_t @-> returning t)

  let cross_out =
    foreign "atg_cross_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let ctc_loss1 =
    foreign "atg_ctc_loss1"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int64_t @-> returning t)

  let ctc_loss2 =
    foreign "atg_ctc_loss2"
    (t @-> t @-> t @-> t @-> int64_t @-> int64_t @-> returning t)

  let cudnn_affine_grid_generator =
    foreign "atg_cudnn_affine_grid_generator"
    (t @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let cudnn_affine_grid_generator_backward =
    foreign "atg_cudnn_affine_grid_generator_backward"
    (t @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let cudnn_convolution =
    foreign "atg_cudnn_convolution"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let cudnn_convolution_backward_bias =
    foreign "atg_cudnn_convolution_backward_bias"
    (t @-> returning t)

  let cudnn_convolution_backward_input =
    foreign "atg_cudnn_convolution_backward_input"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let cudnn_convolution_backward_weight =
    foreign "atg_cudnn_convolution_backward_weight"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let cudnn_convolution_transpose =
    foreign "atg_cudnn_convolution_transpose"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let cudnn_convolution_transpose_backward_bias =
    foreign "atg_cudnn_convolution_transpose_backward_bias"
    (t @-> returning t)

  let cudnn_convolution_transpose_backward_input =
    foreign "atg_cudnn_convolution_transpose_backward_input"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let cudnn_convolution_transpose_backward_weight =
    foreign "atg_cudnn_convolution_transpose_backward_weight"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let cudnn_grid_sampler =
    foreign "atg_cudnn_grid_sampler"
    (t @-> t @-> returning t)

  let cumprod1 =
    foreign "atg_cumprod1"
    (t @-> int64_t @-> int @-> returning t)

  let cumprod2 =
    foreign "atg_cumprod2"
    (t @-> int64_t @-> returning t)

  let cumprod_out1 =
    foreign "atg_cumprod_out1"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let cumprod_out2 =
    foreign "atg_cumprod_out2"
    (t @-> t @-> int64_t @-> returning t)

  let cumsum1 =
    foreign "atg_cumsum1"
    (t @-> int64_t @-> int @-> returning t)

  let cumsum2 =
    foreign "atg_cumsum2"
    (t @-> int64_t @-> returning t)

  let cumsum_out1 =
    foreign "atg_cumsum_out1"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let cumsum_out2 =
    foreign "atg_cumsum_out2"
    (t @-> t @-> int64_t @-> returning t)

  let det =
    foreign "atg_det"
    (t @-> returning t)

  let detach =
    foreign "atg_detach"
    (t @-> returning t)

  let detach_ =
    foreign "atg_detach_"
    (t @-> returning t)

  let diag =
    foreign "atg_diag"
    (t @-> int64_t @-> returning t)

  let diag_out =
    foreign "atg_diag_out"
    (t @-> t @-> int64_t @-> returning t)

  let diagflat =
    foreign "atg_diagflat"
    (t @-> int64_t @-> returning t)

  let diagonal =
    foreign "atg_diagonal"
    (t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let digamma =
    foreign "atg_digamma"
    (t @-> returning t)

  let digamma_ =
    foreign "atg_digamma_"
    (t @-> returning t)

  let digamma_out =
    foreign "atg_digamma_out"
    (t @-> t @-> returning t)

  let div =
    foreign "atg_div"
    (t @-> t @-> returning t)

  let div_ =
    foreign "atg_div_"
    (t @-> t @-> returning t)

  let div_out =
    foreign "atg_div_out"
    (t @-> t @-> t @-> returning t)

  let dot =
    foreign "atg_dot"
    (t @-> t @-> returning t)

  let dot_out =
    foreign "atg_dot_out"
    (t @-> t @-> t @-> returning t)

  let dropout =
    foreign "atg_dropout"
    (t @-> double @-> int @-> returning t)

  let dropout_ =
    foreign "atg_dropout_"
    (t @-> double @-> int @-> returning t)

  let elu =
    foreign "atg_elu"
    (t @-> returning t)

  let elu_ =
    foreign "atg_elu_"
    (t @-> returning t)

  let elu_out =
    foreign "atg_elu_out"
    (t @-> t @-> returning t)

  let empty =
    foreign "atg_empty"
    (ptr int @-> int @-> int @-> returning t)

  let empty_like1 =
    foreign "atg_empty_like1"
    (t @-> returning t)

  let empty_like2 =
    foreign "atg_empty_like2"
    (t @-> int @-> returning t)

  let empty_out =
    foreign "atg_empty_out"
    (t @-> ptr int @-> int @-> returning t)

  let empty_strided =
    foreign "atg_empty_strided"
    (ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let eq =
    foreign "atg_eq"
    (t @-> t @-> returning t)

  let eq_ =
    foreign "atg_eq_"
    (t @-> t @-> returning t)

  let erf =
    foreign "atg_erf"
    (t @-> returning t)

  let erf_ =
    foreign "atg_erf_"
    (t @-> returning t)

  let erf_out =
    foreign "atg_erf_out"
    (t @-> t @-> returning t)

  let erfc =
    foreign "atg_erfc"
    (t @-> returning t)

  let erfc_ =
    foreign "atg_erfc_"
    (t @-> returning t)

  let erfc_out =
    foreign "atg_erfc_out"
    (t @-> t @-> returning t)

  let erfinv =
    foreign "atg_erfinv"
    (t @-> returning t)

  let erfinv_ =
    foreign "atg_erfinv_"
    (t @-> returning t)

  let erfinv_out =
    foreign "atg_erfinv_out"
    (t @-> t @-> returning t)

  let exp =
    foreign "atg_exp"
    (t @-> returning t)

  let exp_ =
    foreign "atg_exp_"
    (t @-> returning t)

  let exp_out =
    foreign "atg_exp_out"
    (t @-> t @-> returning t)

  let expand =
    foreign "atg_expand"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let expand_as =
    foreign "atg_expand_as"
    (t @-> t @-> returning t)

  let expm1 =
    foreign "atg_expm1"
    (t @-> returning t)

  let expm1_ =
    foreign "atg_expm1_"
    (t @-> returning t)

  let expm1_out =
    foreign "atg_expm1_out"
    (t @-> t @-> returning t)

  let exponential_ =
    foreign "atg_exponential_"
    (t @-> double @-> returning t)

  let eye1 =
    foreign "atg_eye1"
    (int64_t @-> int @-> returning t)

  let eye2 =
    foreign "atg_eye2"
    (int64_t @-> int64_t @-> int @-> returning t)

  let eye_out1 =
    foreign "atg_eye_out1"
    (t @-> int64_t @-> returning t)

  let eye_out2 =
    foreign "atg_eye_out2"
    (t @-> int64_t @-> int64_t @-> returning t)

  let feature_alpha_dropout =
    foreign "atg_feature_alpha_dropout"
    (t @-> double @-> int @-> returning t)

  let feature_alpha_dropout_ =
    foreign "atg_feature_alpha_dropout_"
    (t @-> double @-> int @-> returning t)

  let feature_dropout =
    foreign "atg_feature_dropout"
    (t @-> double @-> int @-> returning t)

  let feature_dropout_ =
    foreign "atg_feature_dropout_"
    (t @-> double @-> int @-> returning t)

  let fft =
    foreign "atg_fft"
    (t @-> int64_t @-> int @-> returning t)

  let fill_ =
    foreign "atg_fill_"
    (t @-> t @-> returning t)

  let flatten =
    foreign "atg_flatten"
    (t @-> int64_t @-> int64_t @-> returning t)

  let flip =
    foreign "atg_flip"
    (t @-> ptr int @-> int @-> returning t)

  let floor =
    foreign "atg_floor"
    (t @-> returning t)

  let floor_ =
    foreign "atg_floor_"
    (t @-> returning t)

  let floor_out =
    foreign "atg_floor_out"
    (t @-> t @-> returning t)

  let fmod =
    foreign "atg_fmod"
    (t @-> t @-> returning t)

  let fmod_ =
    foreign "atg_fmod_"
    (t @-> t @-> returning t)

  let fmod_out =
    foreign "atg_fmod_out"
    (t @-> t @-> t @-> returning t)

  let frac =
    foreign "atg_frac"
    (t @-> returning t)

  let frac_ =
    foreign "atg_frac_"
    (t @-> returning t)

  let frac_out =
    foreign "atg_frac_out"
    (t @-> t @-> returning t)

  let frobenius_norm1 =
    foreign "atg_frobenius_norm1"
    (t @-> returning t)

  let frobenius_norm2 =
    foreign "atg_frobenius_norm2"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let frobenius_norm_out =
    foreign "atg_frobenius_norm_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let ge =
    foreign "atg_ge"
    (t @-> t @-> returning t)

  let ge_ =
    foreign "atg_ge_"
    (t @-> t @-> returning t)

  let geometric_ =
    foreign "atg_geometric_"
    (t @-> double @-> returning t)

  let ger =
    foreign "atg_ger"
    (t @-> t @-> returning t)

  let ger_out =
    foreign "atg_ger_out"
    (t @-> t @-> t @-> returning t)

  let glu =
    foreign "atg_glu"
    (t @-> int64_t @-> returning t)

  let glu_backward =
    foreign "atg_glu_backward"
    (t @-> t @-> int64_t @-> returning t)

  let glu_backward_out =
    foreign "atg_glu_backward_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let glu_forward =
    foreign "atg_glu_forward"
    (t @-> int64_t @-> returning t)

  let glu_forward_out =
    foreign "atg_glu_forward_out"
    (t @-> t @-> int64_t @-> returning t)

  let glu_out =
    foreign "atg_glu_out"
    (t @-> t @-> int64_t @-> returning t)

  let grad =
    foreign "atg_grad"
    (t @-> returning t)

  let grid_sampler =
    foreign "atg_grid_sampler"
    (t @-> t @-> int64_t @-> int64_t @-> returning t)

  let grid_sampler_2d =
    foreign "atg_grid_sampler_2d"
    (t @-> t @-> int64_t @-> int64_t @-> returning t)

  let grid_sampler_3d =
    foreign "atg_grid_sampler_3d"
    (t @-> t @-> int64_t @-> int64_t @-> returning t)

  let gru_cell =
    foreign "atg_gru_cell"
    (t @-> t @-> t @-> t @-> t @-> t @-> returning t)

  let gt =
    foreign "atg_gt"
    (t @-> t @-> returning t)

  let gt_ =
    foreign "atg_gt_"
    (t @-> t @-> returning t)

  let hamming_window1 =
    foreign "atg_hamming_window1"
    (int64_t @-> int @-> returning t)

  let hamming_window2 =
    foreign "atg_hamming_window2"
    (int64_t @-> int @-> int @-> returning t)

  let hamming_window3 =
    foreign "atg_hamming_window3"
    (int64_t @-> int @-> double @-> int @-> returning t)

  let hamming_window4 =
    foreign "atg_hamming_window4"
    (int64_t @-> int @-> double @-> double @-> int @-> returning t)

  let hann_window1 =
    foreign "atg_hann_window1"
    (int64_t @-> int @-> returning t)

  let hann_window2 =
    foreign "atg_hann_window2"
    (int64_t @-> int @-> int @-> returning t)

  let hardshrink =
    foreign "atg_hardshrink"
    (t @-> returning t)

  let hardtanh =
    foreign "atg_hardtanh"
    (t @-> returning t)

  let hardtanh_ =
    foreign "atg_hardtanh_"
    (t @-> returning t)

  let hardtanh_out =
    foreign "atg_hardtanh_out"
    (t @-> t @-> returning t)

  let hinge_embedding_loss =
    foreign "atg_hinge_embedding_loss"
    (t @-> t @-> double @-> int64_t @-> returning t)

  let histc =
    foreign "atg_histc"
    (t @-> int64_t @-> returning t)

  let histc_out =
    foreign "atg_histc_out"
    (t @-> t @-> int64_t @-> returning t)

  let hspmm =
    foreign "atg_hspmm"
    (t @-> t @-> returning t)

  let hspmm_out =
    foreign "atg_hspmm_out"
    (t @-> t @-> t @-> returning t)

  let ifft =
    foreign "atg_ifft"
    (t @-> int64_t @-> int @-> returning t)

  let instance_norm =
    foreign "atg_instance_norm"
    (t @-> t @-> t @-> t @-> t @-> int @-> double @-> double @-> int @-> returning t)

  let inverse =
    foreign "atg_inverse"
    (t @-> returning t)

  let inverse_out =
    foreign "atg_inverse_out"
    (t @-> t @-> returning t)

  let irfft =
    foreign "atg_irfft"
    (t @-> int64_t @-> int @-> int @-> ptr int @-> int @-> returning t)

  let isclose =
    foreign "atg_isclose"
    (t @-> t @-> double @-> double @-> int @-> returning t)

  let kl_div =
    foreign "atg_kl_div"
    (t @-> t @-> int64_t @-> returning t)

  let kl_div_backward =
    foreign "atg_kl_div_backward"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let l1_loss =
    foreign "atg_l1_loss"
    (t @-> t @-> int64_t @-> returning t)

  let l1_loss_backward =
    foreign "atg_l1_loss_backward"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let l1_loss_backward_out =
    foreign "atg_l1_loss_backward_out"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let l1_loss_forward =
    foreign "atg_l1_loss_forward"
    (t @-> t @-> int64_t @-> returning t)

  let l1_loss_forward_out =
    foreign "atg_l1_loss_forward_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let l1_loss_out =
    foreign "atg_l1_loss_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let le =
    foreign "atg_le"
    (t @-> t @-> returning t)

  let le_ =
    foreign "atg_le_"
    (t @-> t @-> returning t)

  let leaky_relu =
    foreign "atg_leaky_relu"
    (t @-> returning t)

  let leaky_relu_ =
    foreign "atg_leaky_relu_"
    (t @-> returning t)

  let leaky_relu_out =
    foreign "atg_leaky_relu_out"
    (t @-> t @-> returning t)

  let lgamma =
    foreign "atg_lgamma"
    (t @-> returning t)

  let lgamma_ =
    foreign "atg_lgamma_"
    (t @-> returning t)

  let lgamma_out =
    foreign "atg_lgamma_out"
    (t @-> t @-> returning t)

  let linear =
    foreign "atg_linear"
    (t @-> t @-> t @-> returning t)

  let log =
    foreign "atg_log"
    (t @-> returning t)

  let log10 =
    foreign "atg_log10"
    (t @-> returning t)

  let log10_ =
    foreign "atg_log10_"
    (t @-> returning t)

  let log10_out =
    foreign "atg_log10_out"
    (t @-> t @-> returning t)

  let log1p =
    foreign "atg_log1p"
    (t @-> returning t)

  let log1p_ =
    foreign "atg_log1p_"
    (t @-> returning t)

  let log1p_out =
    foreign "atg_log1p_out"
    (t @-> t @-> returning t)

  let log2 =
    foreign "atg_log2"
    (t @-> returning t)

  let log2_ =
    foreign "atg_log2_"
    (t @-> returning t)

  let log2_out =
    foreign "atg_log2_out"
    (t @-> t @-> returning t)

  let log_ =
    foreign "atg_log_"
    (t @-> returning t)

  let log_normal_ =
    foreign "atg_log_normal_"
    (t @-> double @-> double @-> returning t)

  let log_out =
    foreign "atg_log_out"
    (t @-> t @-> returning t)

  let log_sigmoid =
    foreign "atg_log_sigmoid"
    (t @-> returning t)

  let log_sigmoid_backward =
    foreign "atg_log_sigmoid_backward"
    (t @-> t @-> t @-> returning t)

  let log_sigmoid_backward_out =
    foreign "atg_log_sigmoid_backward_out"
    (t @-> t @-> t @-> t @-> returning t)

  let log_sigmoid_out =
    foreign "atg_log_sigmoid_out"
    (t @-> t @-> returning t)

  let log_softmax =
    foreign "atg_log_softmax"
    (t @-> int64_t @-> returning t)

  let log_softmax_backward_data =
    foreign "atg_log_softmax_backward_data"
    (t @-> t @-> int64_t @-> t @-> returning t)

  let logdet =
    foreign "atg_logdet"
    (t @-> returning t)

  let logsumexp =
    foreign "atg_logsumexp"
    (t @-> int64_t @-> int @-> returning t)

  let logsumexp_out =
    foreign "atg_logsumexp_out"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let lt =
    foreign "atg_lt"
    (t @-> t @-> returning t)

  let lt_ =
    foreign "atg_lt_"
    (t @-> t @-> returning t)

  let margin_ranking_loss =
    foreign "atg_margin_ranking_loss"
    (t @-> t @-> t @-> double @-> int64_t @-> returning t)

  let matmul =
    foreign "atg_matmul"
    (t @-> t @-> returning t)

  let matmul_out =
    foreign "atg_matmul_out"
    (t @-> t @-> t @-> returning t)

  let matrix_power =
    foreign "atg_matrix_power"
    (t @-> int64_t @-> returning t)

  let matrix_rank1 =
    foreign "atg_matrix_rank1"
    (t @-> double @-> int @-> returning t)

  let matrix_rank2 =
    foreign "atg_matrix_rank2"
    (t @-> int @-> returning t)

  let max =
    foreign "atg_max"
    (t @-> t @-> returning t)

  let max_out =
    foreign "atg_max_out"
    (t @-> t @-> t @-> returning t)

  let max_pool1d =
    foreign "atg_max_pool1d"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let max_pool2d =
    foreign "atg_max_pool2d"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let max_pool3d =
    foreign "atg_max_pool3d"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let max_values =
    foreign "atg_max_values"
    (t @-> int64_t @-> int @-> returning t)

  let mean1 =
    foreign "atg_mean1"
    (t @-> int @-> returning t)

  let mean2 =
    foreign "atg_mean2"
    (t @-> returning t)

  let mean3 =
    foreign "atg_mean3"
    (t @-> int64_t @-> int @-> int @-> returning t)

  let mean4 =
    foreign "atg_mean4"
    (t @-> int64_t @-> int @-> returning t)

  let mean5 =
    foreign "atg_mean5"
    (t @-> int64_t @-> int @-> returning t)

  let mean_out1 =
    foreign "atg_mean_out1"
    (t @-> t @-> int64_t @-> int @-> int @-> returning t)

  let mean_out2 =
    foreign "atg_mean_out2"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let mean_out3 =
    foreign "atg_mean_out3"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let min =
    foreign "atg_min"
    (t @-> t @-> returning t)

  let min_out =
    foreign "atg_min_out"
    (t @-> t @-> t @-> returning t)

  let min_values =
    foreign "atg_min_values"
    (t @-> int64_t @-> int @-> returning t)

  let miopen_convolution =
    foreign "atg_miopen_convolution"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let miopen_convolution_backward_bias =
    foreign "atg_miopen_convolution_backward_bias"
    (t @-> returning t)

  let miopen_convolution_backward_input =
    foreign "atg_miopen_convolution_backward_input"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let miopen_convolution_backward_weight =
    foreign "atg_miopen_convolution_backward_weight"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let miopen_convolution_transpose =
    foreign "atg_miopen_convolution_transpose"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let miopen_convolution_transpose_backward_input =
    foreign "atg_miopen_convolution_transpose_backward_input"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let miopen_convolution_transpose_backward_weight =
    foreign "atg_miopen_convolution_transpose_backward_weight"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> int @-> returning t)

  let mkldnn_convolution =
    foreign "atg_mkldnn_convolution"
    (t @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> returning t)

  let mkldnn_convolution_backward_input =
    foreign "atg_mkldnn_convolution_backward_input"
    (ptr int @-> int @-> t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> int64_t @-> int @-> returning t)

  let mm =
    foreign "atg_mm"
    (t @-> t @-> returning t)

  let mm_out =
    foreign "atg_mm_out"
    (t @-> t @-> t @-> returning t)

  let mse_loss =
    foreign "atg_mse_loss"
    (t @-> t @-> int64_t @-> returning t)

  let mse_loss_backward =
    foreign "atg_mse_loss_backward"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let mse_loss_backward_out =
    foreign "atg_mse_loss_backward_out"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let mse_loss_forward =
    foreign "atg_mse_loss_forward"
    (t @-> t @-> int64_t @-> returning t)

  let mse_loss_forward_out =
    foreign "atg_mse_loss_forward_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let mse_loss_out =
    foreign "atg_mse_loss_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let mul =
    foreign "atg_mul"
    (t @-> t @-> returning t)

  let mul_ =
    foreign "atg_mul_"
    (t @-> t @-> returning t)

  let mul_out =
    foreign "atg_mul_out"
    (t @-> t @-> t @-> returning t)

  let mv =
    foreign "atg_mv"
    (t @-> t @-> returning t)

  let mv_out =
    foreign "atg_mv_out"
    (t @-> t @-> t @-> returning t)

  let mvlgamma =
    foreign "atg_mvlgamma"
    (t @-> int64_t @-> returning t)

  let mvlgamma_ =
    foreign "atg_mvlgamma_"
    (t @-> int64_t @-> returning t)

  let narrow =
    foreign "atg_narrow"
    (t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let narrow_copy =
    foreign "atg_narrow_copy"
    (t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let native_clone =
    foreign "atg_native_clone"
    (t @-> returning t)

  let native_norm =
    foreign "atg_native_norm"
    (t @-> returning t)

  let native_resize_as_ =
    foreign "atg_native_resize_as_"
    (t @-> t @-> returning t)

  let native_zero_ =
    foreign "atg_native_zero_"
    (t @-> returning t)

  let ne =
    foreign "atg_ne"
    (t @-> t @-> returning t)

  let ne_ =
    foreign "atg_ne_"
    (t @-> t @-> returning t)

  let neg =
    foreign "atg_neg"
    (t @-> returning t)

  let neg_ =
    foreign "atg_neg_"
    (t @-> returning t)

  let neg_out =
    foreign "atg_neg_out"
    (t @-> t @-> returning t)

  let norm =
    foreign "atg_norm"
    (t @-> returning t)

  let norm_except_dim =
    foreign "atg_norm_except_dim"
    (t @-> int64_t @-> int64_t @-> returning t)

  let normal1 =
    foreign "atg_normal1"
    (t @-> double @-> returning t)

  let normal2 =
    foreign "atg_normal2"
    (double @-> t @-> returning t)

  let normal3 =
    foreign "atg_normal3"
    (t @-> t @-> returning t)

  let normal_ =
    foreign "atg_normal_"
    (t @-> double @-> double @-> returning t)

  let normal_out1 =
    foreign "atg_normal_out1"
    (t @-> t @-> double @-> returning t)

  let normal_out2 =
    foreign "atg_normal_out2"
    (t @-> double @-> t @-> returning t)

  let normal_out3 =
    foreign "atg_normal_out3"
    (t @-> t @-> t @-> returning t)

  let nuclear_norm =
    foreign "atg_nuclear_norm"
    (t @-> int @-> returning t)

  let nuclear_norm_out =
    foreign "atg_nuclear_norm_out"
    (t @-> t @-> int @-> returning t)

  let ones =
    foreign "atg_ones"
    (ptr int @-> int @-> int @-> returning t)

  let ones_like1 =
    foreign "atg_ones_like1"
    (t @-> returning t)

  let ones_like2 =
    foreign "atg_ones_like2"
    (t @-> int @-> returning t)

  let ones_out =
    foreign "atg_ones_out"
    (t @-> ptr int @-> int @-> returning t)

  let orgqr =
    foreign "atg_orgqr"
    (t @-> t @-> returning t)

  let orgqr_out =
    foreign "atg_orgqr_out"
    (t @-> t @-> t @-> returning t)

  let ormqr =
    foreign "atg_ormqr"
    (t @-> t @-> t @-> int @-> int @-> returning t)

  let ormqr_out =
    foreign "atg_ormqr_out"
    (t @-> t @-> t @-> t @-> int @-> int @-> returning t)

  let pairwise_distance =
    foreign "atg_pairwise_distance"
    (t @-> t @-> double @-> double @-> int @-> returning t)

  let pdist =
    foreign "atg_pdist"
    (t @-> double @-> returning t)

  let permute =
    foreign "atg_permute"
    (t @-> ptr int @-> int @-> returning t)

  let pin_memory =
    foreign "atg_pin_memory"
    (t @-> returning t)

  let pinverse =
    foreign "atg_pinverse"
    (t @-> double @-> returning t)

  let pixel_shuffle =
    foreign "atg_pixel_shuffle"
    (t @-> int64_t @-> returning t)

  let poisson =
    foreign "atg_poisson"
    (t @-> returning t)

  let polygamma =
    foreign "atg_polygamma"
    (int64_t @-> t @-> returning t)

  let polygamma_ =
    foreign "atg_polygamma_"
    (t @-> int64_t @-> returning t)

  let polygamma_out =
    foreign "atg_polygamma_out"
    (t @-> int64_t @-> t @-> returning t)

  let potrf =
    foreign "atg_potrf"
    (t @-> int @-> returning t)

  let potrf_out =
    foreign "atg_potrf_out"
    (t @-> t @-> int @-> returning t)

  let potri =
    foreign "atg_potri"
    (t @-> int @-> returning t)

  let potri_out =
    foreign "atg_potri_out"
    (t @-> t @-> int @-> returning t)

  let potrs =
    foreign "atg_potrs"
    (t @-> t @-> int @-> returning t)

  let potrs_out =
    foreign "atg_potrs_out"
    (t @-> t @-> t @-> int @-> returning t)

  let pow =
    foreign "atg_pow"
    (t @-> t @-> returning t)

  let pow_ =
    foreign "atg_pow_"
    (t @-> t @-> returning t)

  let pow_out =
    foreign "atg_pow_out"
    (t @-> t @-> t @-> returning t)

  let prelu =
    foreign "atg_prelu"
    (t @-> t @-> returning t)

  let prod1 =
    foreign "atg_prod1"
    (t @-> int @-> returning t)

  let prod2 =
    foreign "atg_prod2"
    (t @-> returning t)

  let prod3 =
    foreign "atg_prod3"
    (t @-> int64_t @-> int @-> int @-> returning t)

  let prod4 =
    foreign "atg_prod4"
    (t @-> int64_t @-> int @-> returning t)

  let prod5 =
    foreign "atg_prod5"
    (t @-> int64_t @-> int @-> returning t)

  let prod_out1 =
    foreign "atg_prod_out1"
    (t @-> t @-> int64_t @-> int @-> int @-> returning t)

  let prod_out2 =
    foreign "atg_prod_out2"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let prod_out3 =
    foreign "atg_prod_out3"
    (t @-> t @-> int64_t @-> int @-> returning t)

  let rand =
    foreign "atg_rand"
    (ptr int @-> int @-> int @-> returning t)

  let rand_like1 =
    foreign "atg_rand_like1"
    (t @-> returning t)

  let rand_like2 =
    foreign "atg_rand_like2"
    (t @-> int @-> returning t)

  let rand_out =
    foreign "atg_rand_out"
    (t @-> ptr int @-> int @-> returning t)

  let randint1 =
    foreign "atg_randint1"
    (int64_t @-> ptr int @-> int @-> int @-> returning t)

  let randint2 =
    foreign "atg_randint2"
    (int64_t @-> int64_t @-> ptr int @-> int @-> int @-> returning t)

  let randint_like1 =
    foreign "atg_randint_like1"
    (t @-> int64_t @-> returning t)

  let randint_like2 =
    foreign "atg_randint_like2"
    (t @-> int64_t @-> int64_t @-> returning t)

  let randint_like3 =
    foreign "atg_randint_like3"
    (t @-> int64_t @-> int @-> returning t)

  let randint_like4 =
    foreign "atg_randint_like4"
    (t @-> int64_t @-> int64_t @-> int @-> returning t)

  let randint_out1 =
    foreign "atg_randint_out1"
    (t @-> int64_t @-> ptr int @-> int @-> returning t)

  let randint_out2 =
    foreign "atg_randint_out2"
    (t @-> int64_t @-> int64_t @-> ptr int @-> int @-> returning t)

  let randn =
    foreign "atg_randn"
    (ptr int @-> int @-> int @-> returning t)

  let randn_like1 =
    foreign "atg_randn_like1"
    (t @-> returning t)

  let randn_like2 =
    foreign "atg_randn_like2"
    (t @-> int @-> returning t)

  let randn_out =
    foreign "atg_randn_out"
    (t @-> ptr int @-> int @-> returning t)

  let random_1 =
    foreign "atg_random_1"
    (t @-> int64_t @-> int64_t @-> returning t)

  let random_2 =
    foreign "atg_random_2"
    (t @-> int64_t @-> returning t)

  let random_3 =
    foreign "atg_random_3"
    (t @-> returning t)

  let randperm =
    foreign "atg_randperm"
    (int64_t @-> int @-> returning t)

  let randperm_out =
    foreign "atg_randperm_out"
    (t @-> int64_t @-> returning t)

  let reciprocal =
    foreign "atg_reciprocal"
    (t @-> returning t)

  let reciprocal_ =
    foreign "atg_reciprocal_"
    (t @-> returning t)

  let reciprocal_out =
    foreign "atg_reciprocal_out"
    (t @-> t @-> returning t)

  let reflection_pad1d =
    foreign "atg_reflection_pad1d"
    (t @-> ptr int @-> int @-> returning t)

  let reflection_pad1d_backward =
    foreign "atg_reflection_pad1d_backward"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad1d_backward_out =
    foreign "atg_reflection_pad1d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad1d_forward =
    foreign "atg_reflection_pad1d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let reflection_pad1d_forward_out =
    foreign "atg_reflection_pad1d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad1d_out =
    foreign "atg_reflection_pad1d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad2d =
    foreign "atg_reflection_pad2d"
    (t @-> ptr int @-> int @-> returning t)

  let reflection_pad2d_backward =
    foreign "atg_reflection_pad2d_backward"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad2d_backward_out =
    foreign "atg_reflection_pad2d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad2d_forward =
    foreign "atg_reflection_pad2d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let reflection_pad2d_forward_out =
    foreign "atg_reflection_pad2d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let reflection_pad2d_out =
    foreign "atg_reflection_pad2d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let relu =
    foreign "atg_relu"
    (t @-> returning t)

  let relu_ =
    foreign "atg_relu_"
    (t @-> returning t)

  let remainder =
    foreign "atg_remainder"
    (t @-> t @-> returning t)

  let remainder_ =
    foreign "atg_remainder_"
    (t @-> t @-> returning t)

  let remainder_out =
    foreign "atg_remainder_out"
    (t @-> t @-> t @-> returning t)

  let repeat =
    foreign "atg_repeat"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad1d =
    foreign "atg_replication_pad1d"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad1d_backward =
    foreign "atg_replication_pad1d_backward"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad1d_backward_out =
    foreign "atg_replication_pad1d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad1d_forward =
    foreign "atg_replication_pad1d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad1d_forward_out =
    foreign "atg_replication_pad1d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad1d_out =
    foreign "atg_replication_pad1d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad2d =
    foreign "atg_replication_pad2d"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad2d_backward =
    foreign "atg_replication_pad2d_backward"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad2d_backward_out =
    foreign "atg_replication_pad2d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad2d_forward =
    foreign "atg_replication_pad2d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad2d_forward_out =
    foreign "atg_replication_pad2d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad2d_out =
    foreign "atg_replication_pad2d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad3d =
    foreign "atg_replication_pad3d"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad3d_backward =
    foreign "atg_replication_pad3d_backward"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad3d_backward_out =
    foreign "atg_replication_pad3d_backward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad3d_forward =
    foreign "atg_replication_pad3d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let replication_pad3d_forward_out =
    foreign "atg_replication_pad3d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let replication_pad3d_out =
    foreign "atg_replication_pad3d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let reshape =
    foreign "atg_reshape"
    (t @-> ptr int @-> int @-> returning t)

  let reshape_as =
    foreign "atg_reshape_as"
    (t @-> t @-> returning t)

  let resize_ =
    foreign "atg_resize_"
    (t @-> ptr int @-> int @-> returning t)

  let resize_as_ =
    foreign "atg_resize_as_"
    (t @-> t @-> returning t)

  let rfft =
    foreign "atg_rfft"
    (t @-> int64_t @-> int @-> int @-> returning t)

  let rnn_relu_cell =
    foreign "atg_rnn_relu_cell"
    (t @-> t @-> t @-> t @-> t @-> t @-> returning t)

  let rnn_tanh_cell =
    foreign "atg_rnn_tanh_cell"
    (t @-> t @-> t @-> t @-> t @-> t @-> returning t)

  let roipooling2d_backward =
    foreign "atg_roipooling2d_backward"
    (t @-> t @-> int64_t @-> int64_t @-> double @-> t @-> t @-> returning t)

  let round =
    foreign "atg_round"
    (t @-> returning t)

  let round_ =
    foreign "atg_round_"
    (t @-> returning t)

  let round_out =
    foreign "atg_round_out"
    (t @-> t @-> returning t)

  let rrelu =
    foreign "atg_rrelu"
    (t @-> int @-> returning t)

  let rrelu_ =
    foreign "atg_rrelu_"
    (t @-> int @-> returning t)

  let rrelu_with_noise =
    foreign "atg_rrelu_with_noise"
    (t @-> t @-> int @-> returning t)

  let rrelu_with_noise_ =
    foreign "atg_rrelu_with_noise_"
    (t @-> t @-> int @-> returning t)

  let rrelu_with_noise_out =
    foreign "atg_rrelu_with_noise_out"
    (t @-> t @-> t @-> int @-> returning t)

  let rsqrt =
    foreign "atg_rsqrt"
    (t @-> returning t)

  let rsqrt_ =
    foreign "atg_rsqrt_"
    (t @-> returning t)

  let rsqrt_out =
    foreign "atg_rsqrt_out"
    (t @-> t @-> returning t)

  let s_native_addmm =
    foreign "atg_s_native_addmm"
    (t @-> t @-> t @-> returning t)

  let s_native_addmm_ =
    foreign "atg_s_native_addmm_"
    (t @-> t @-> t @-> returning t)

  let s_native_addmm_out =
    foreign "atg_s_native_addmm_out"
    (t @-> t @-> t @-> t @-> returning t)

  let select =
    foreign "atg_select"
    (t @-> int64_t @-> int64_t @-> returning t)

  let selu =
    foreign "atg_selu"
    (t @-> returning t)

  let selu_ =
    foreign "atg_selu_"
    (t @-> returning t)

  let set_1 =
    foreign "atg_set_1"
    (t @-> t @-> returning t)

  let set_2 =
    foreign "atg_set_2"
    (t @-> returning t)

  let set_requires_grad =
    foreign "atg_set_requires_grad"
    (t @-> int @-> returning t)

  let sigmoid =
    foreign "atg_sigmoid"
    (t @-> returning t)

  let sigmoid_ =
    foreign "atg_sigmoid_"
    (t @-> returning t)

  let sigmoid_out =
    foreign "atg_sigmoid_out"
    (t @-> t @-> returning t)

  let sign =
    foreign "atg_sign"
    (t @-> returning t)

  let sign_ =
    foreign "atg_sign_"
    (t @-> returning t)

  let sign_out =
    foreign "atg_sign_out"
    (t @-> t @-> returning t)

  let sin =
    foreign "atg_sin"
    (t @-> returning t)

  let sin_ =
    foreign "atg_sin_"
    (t @-> returning t)

  let sin_out =
    foreign "atg_sin_out"
    (t @-> t @-> returning t)

  let sinh =
    foreign "atg_sinh"
    (t @-> returning t)

  let sinh_ =
    foreign "atg_sinh_"
    (t @-> returning t)

  let sinh_out =
    foreign "atg_sinh_out"
    (t @-> t @-> returning t)

  let slice =
    foreign "atg_slice"
    (t @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let smm =
    foreign "atg_smm"
    (t @-> t @-> returning t)

  let smooth_l1_loss =
    foreign "atg_smooth_l1_loss"
    (t @-> t @-> int64_t @-> returning t)

  let smooth_l1_loss_backward =
    foreign "atg_smooth_l1_loss_backward"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let smooth_l1_loss_backward_out =
    foreign "atg_smooth_l1_loss_backward_out"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let smooth_l1_loss_forward =
    foreign "atg_smooth_l1_loss_forward"
    (t @-> t @-> int64_t @-> returning t)

  let smooth_l1_loss_forward_out =
    foreign "atg_smooth_l1_loss_forward_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let smooth_l1_loss_out =
    foreign "atg_smooth_l1_loss_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let soft_margin_loss =
    foreign "atg_soft_margin_loss"
    (t @-> t @-> int64_t @-> returning t)

  let soft_margin_loss_backward =
    foreign "atg_soft_margin_loss_backward"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let soft_margin_loss_backward_out =
    foreign "atg_soft_margin_loss_backward_out"
    (t @-> t @-> t @-> t @-> int64_t @-> returning t)

  let soft_margin_loss_forward =
    foreign "atg_soft_margin_loss_forward"
    (t @-> t @-> int64_t @-> returning t)

  let soft_margin_loss_forward_out =
    foreign "atg_soft_margin_loss_forward_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let soft_margin_loss_out =
    foreign "atg_soft_margin_loss_out"
    (t @-> t @-> t @-> int64_t @-> returning t)

  let softmax =
    foreign "atg_softmax"
    (t @-> int64_t @-> returning t)

  let softmax_backward_data =
    foreign "atg_softmax_backward_data"
    (t @-> t @-> int64_t @-> t @-> returning t)

  let softplus =
    foreign "atg_softplus"
    (t @-> returning t)

  let softplus_out =
    foreign "atg_softplus_out"
    (t @-> t @-> returning t)

  let softshrink =
    foreign "atg_softshrink"
    (t @-> returning t)

  let softshrink_out =
    foreign "atg_softshrink_out"
    (t @-> t @-> returning t)

  let sparse_coo_tensor =
    foreign "atg_sparse_coo_tensor"
    (ptr int @-> int @-> int @-> returning t)

  let sparse_resize_ =
    foreign "atg_sparse_resize_"
    (t @-> ptr int @-> int @-> int64_t @-> int64_t @-> returning t)

  let sparse_resize_and_clear_ =
    foreign "atg_sparse_resize_and_clear_"
    (t @-> ptr int @-> int @-> int64_t @-> int64_t @-> returning t)

  let sqrt =
    foreign "atg_sqrt"
    (t @-> returning t)

  let sqrt_ =
    foreign "atg_sqrt_"
    (t @-> returning t)

  let sqrt_out =
    foreign "atg_sqrt_out"
    (t @-> t @-> returning t)

  let squeeze1 =
    foreign "atg_squeeze1"
    (t @-> returning t)

  let squeeze2 =
    foreign "atg_squeeze2"
    (t @-> int64_t @-> returning t)

  let squeeze_1 =
    foreign "atg_squeeze_1"
    (t @-> returning t)

  let squeeze_2 =
    foreign "atg_squeeze_2"
    (t @-> int64_t @-> returning t)

  let sspaddmm =
    foreign "atg_sspaddmm"
    (t @-> t @-> t @-> returning t)

  let sspaddmm_out =
    foreign "atg_sspaddmm_out"
    (t @-> t @-> t @-> t @-> returning t)

  let std1 =
    foreign "atg_std1"
    (t @-> int @-> returning t)

  let std2 =
    foreign "atg_std2"
    (t @-> int64_t @-> int @-> int @-> returning t)

  let std_out =
    foreign "atg_std_out"
    (t @-> t @-> int64_t @-> int @-> int @-> returning t)

  let sub =
    foreign "atg_sub"
    (t @-> t @-> returning t)

  let sub_ =
    foreign "atg_sub_"
    (t @-> t @-> returning t)

  let sub_out =
    foreign "atg_sub_out"
    (t @-> t @-> t @-> returning t)

  let sum1 =
    foreign "atg_sum1"
    (t @-> int @-> returning t)

  let sum2 =
    foreign "atg_sum2"
    (t @-> returning t)

  let sum3 =
    foreign "atg_sum3"
    (t @-> ptr int @-> int @-> int @-> int @-> returning t)

  let sum4 =
    foreign "atg_sum4"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let sum5 =
    foreign "atg_sum5"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let sum_out1 =
    foreign "atg_sum_out1"
    (t @-> t @-> ptr int @-> int @-> int @-> int @-> returning t)

  let sum_out2 =
    foreign "atg_sum_out2"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let sum_out3 =
    foreign "atg_sum_out3"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let t_ =
    foreign "atg_t_"
    (t @-> returning t)

  let tan =
    foreign "atg_tan"
    (t @-> returning t)

  let tan_ =
    foreign "atg_tan_"
    (t @-> returning t)

  let tan_out =
    foreign "atg_tan_out"
    (t @-> t @-> returning t)

  let tanh =
    foreign "atg_tanh"
    (t @-> returning t)

  let tanh_ =
    foreign "atg_tanh_"
    (t @-> returning t)

  let tanh_out =
    foreign "atg_tanh_out"
    (t @-> t @-> returning t)

  let tensordot =
    foreign "atg_tensordot"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let th_addmm =
    foreign "atg_th_addmm"
    (t @-> t @-> t @-> returning t)

  let th_addmm_ =
    foreign "atg_th_addmm_"
    (t @-> t @-> t @-> returning t)

  let th_addmm_out =
    foreign "atg_th_addmm_out"
    (t @-> t @-> t @-> t @-> returning t)

  let th_clone =
    foreign "atg_th_clone"
    (t @-> returning t)

  let th_resize_as_ =
    foreign "atg_th_resize_as_"
    (t @-> t @-> returning t)

  let th_zero_ =
    foreign "atg_th_zero_"
    (t @-> returning t)

  let thnn_batch_norm =
    foreign "atg_thnn_batch_norm"
    (t @-> t @-> t @-> t @-> t @-> int @-> double @-> double @-> returning t)

  let thnn_batch_norm_out =
    foreign "atg_thnn_batch_norm_out"
    (t @-> t @-> t @-> t @-> t @-> t @-> int @-> double @-> double @-> returning t)

  let thnn_conv2d =
    foreign "atg_thnn_conv2d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv2d_out =
    foreign "atg_thnn_conv2d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv3d =
    foreign "atg_thnn_conv3d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv3d_out =
    foreign "atg_thnn_conv3d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_depthwise2d =
    foreign "atg_thnn_conv_depthwise2d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_depthwise2d_forward =
    foreign "atg_thnn_conv_depthwise2d_forward"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_depthwise2d_forward_out =
    foreign "atg_thnn_conv_depthwise2d_forward_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_depthwise2d_out =
    foreign "atg_thnn_conv_depthwise2d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_dilated2d =
    foreign "atg_thnn_conv_dilated2d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_dilated2d_out =
    foreign "atg_thnn_conv_dilated2d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_dilated3d =
    foreign "atg_thnn_conv_dilated3d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_dilated3d_out =
    foreign "atg_thnn_conv_dilated3d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_transpose2d =
    foreign "atg_thnn_conv_transpose2d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_transpose2d_out =
    foreign "atg_thnn_conv_transpose2d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_transpose3d =
    foreign "atg_thnn_conv_transpose3d"
    (t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let thnn_conv_transpose3d_out =
    foreign "atg_thnn_conv_transpose3d_out"
    (t @-> t @-> t @-> ptr int @-> int @-> t @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let to1 =
    foreign "atg_to1"
    (t @-> int @-> returning t)

  let to2 =
    foreign "atg_to2"
    (t @-> int @-> int @-> int @-> returning t)

  let to3 =
    foreign "atg_to3"
    (t @-> int @-> int @-> returning t)

  let to4 =
    foreign "atg_to4"
    (t @-> int @-> int @-> returning t)

  let to5 =
    foreign "atg_to5"
    (t @-> t @-> int @-> returning t)

  let to_dense =
    foreign "atg_to_dense"
    (t @-> returning t)

  let totype =
    foreign "atg_totype"
    (t @-> int @-> returning t)

  let transpose =
    foreign "atg_transpose"
    (t @-> int64_t @-> int64_t @-> returning t)

  let transpose_ =
    foreign "atg_transpose_"
    (t @-> int64_t @-> int64_t @-> returning t)

  let tril =
    foreign "atg_tril"
    (t @-> int64_t @-> returning t)

  let tril_ =
    foreign "atg_tril_"
    (t @-> int64_t @-> returning t)

  let tril_out =
    foreign "atg_tril_out"
    (t @-> t @-> int64_t @-> returning t)

  let triplet_margin_loss =
    foreign "atg_triplet_margin_loss"
    (t @-> t @-> t @-> double @-> double @-> double @-> int @-> int64_t @-> returning t)

  let triu =
    foreign "atg_triu"
    (t @-> int64_t @-> returning t)

  let triu_ =
    foreign "atg_triu_"
    (t @-> int64_t @-> returning t)

  let triu_out =
    foreign "atg_triu_out"
    (t @-> t @-> int64_t @-> returning t)

  let trunc =
    foreign "atg_trunc"
    (t @-> returning t)

  let trunc_ =
    foreign "atg_trunc_"
    (t @-> returning t)

  let trunc_out =
    foreign "atg_trunc_out"
    (t @-> t @-> returning t)

  let type_as =
    foreign "atg_type_as"
    (t @-> t @-> returning t)

  let unfold =
    foreign "atg_unfold"
    (t @-> int64_t @-> int64_t @-> int64_t @-> returning t)

  let uniform_ =
    foreign "atg_uniform_"
    (t @-> double @-> double @-> returning t)

  let unsqueeze =
    foreign "atg_unsqueeze"
    (t @-> int64_t @-> returning t)

  let unsqueeze_ =
    foreign "atg_unsqueeze_"
    (t @-> int64_t @-> returning t)

  let upsample_bilinear2d =
    foreign "atg_upsample_bilinear2d"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_bilinear2d_backward =
    foreign "atg_upsample_bilinear2d_backward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let upsample_bilinear2d_backward_out =
    foreign "atg_upsample_bilinear2d_backward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let upsample_bilinear2d_forward =
    foreign "atg_upsample_bilinear2d_forward"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_bilinear2d_forward_out =
    foreign "atg_upsample_bilinear2d_forward_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_bilinear2d_out =
    foreign "atg_upsample_bilinear2d_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_linear1d =
    foreign "atg_upsample_linear1d"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_linear1d_backward =
    foreign "atg_upsample_linear1d_backward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let upsample_linear1d_backward_out =
    foreign "atg_upsample_linear1d_backward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let upsample_linear1d_forward =
    foreign "atg_upsample_linear1d_forward"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_linear1d_forward_out =
    foreign "atg_upsample_linear1d_forward_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_linear1d_out =
    foreign "atg_upsample_linear1d_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_nearest1d =
    foreign "atg_upsample_nearest1d"
    (t @-> ptr int @-> int @-> returning t)

  let upsample_nearest1d_backward =
    foreign "atg_upsample_nearest1d_backward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let upsample_nearest1d_backward_out =
    foreign "atg_upsample_nearest1d_backward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let upsample_nearest1d_forward =
    foreign "atg_upsample_nearest1d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let upsample_nearest1d_forward_out =
    foreign "atg_upsample_nearest1d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let upsample_nearest1d_out =
    foreign "atg_upsample_nearest1d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let upsample_nearest2d =
    foreign "atg_upsample_nearest2d"
    (t @-> ptr int @-> int @-> returning t)

  let upsample_nearest2d_backward =
    foreign "atg_upsample_nearest2d_backward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let upsample_nearest2d_backward_out =
    foreign "atg_upsample_nearest2d_backward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let upsample_nearest2d_forward =
    foreign "atg_upsample_nearest2d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let upsample_nearest2d_forward_out =
    foreign "atg_upsample_nearest2d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let upsample_nearest2d_out =
    foreign "atg_upsample_nearest2d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let upsample_nearest3d =
    foreign "atg_upsample_nearest3d"
    (t @-> ptr int @-> int @-> returning t)

  let upsample_nearest3d_backward =
    foreign "atg_upsample_nearest3d_backward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let upsample_nearest3d_backward_out =
    foreign "atg_upsample_nearest3d_backward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> returning t)

  let upsample_nearest3d_forward =
    foreign "atg_upsample_nearest3d_forward"
    (t @-> ptr int @-> int @-> returning t)

  let upsample_nearest3d_forward_out =
    foreign "atg_upsample_nearest3d_forward_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let upsample_nearest3d_out =
    foreign "atg_upsample_nearest3d_out"
    (t @-> t @-> ptr int @-> int @-> returning t)

  let upsample_trilinear3d =
    foreign "atg_upsample_trilinear3d"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_trilinear3d_backward =
    foreign "atg_upsample_trilinear3d_backward"
    (t @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let upsample_trilinear3d_backward_out =
    foreign "atg_upsample_trilinear3d_backward_out"
    (t @-> t @-> ptr int @-> int @-> ptr int @-> int @-> int @-> returning t)

  let upsample_trilinear3d_forward =
    foreign "atg_upsample_trilinear3d_forward"
    (t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_trilinear3d_forward_out =
    foreign "atg_upsample_trilinear3d_forward_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let upsample_trilinear3d_out =
    foreign "atg_upsample_trilinear3d_out"
    (t @-> t @-> ptr int @-> int @-> int @-> returning t)

  let var1 =
    foreign "atg_var1"
    (t @-> int @-> returning t)

  let var2 =
    foreign "atg_var2"
    (t @-> int64_t @-> int @-> int @-> returning t)

  let var_out =
    foreign "atg_var_out"
    (t @-> t @-> int64_t @-> int @-> int @-> returning t)

  let view =
    foreign "atg_view"
    (t @-> ptr int @-> int @-> returning t)

  let view_as =
    foreign "atg_view_as"
    (t @-> t @-> returning t)

  let zero_ =
    foreign "atg_zero_"
    (t @-> returning t)

  let zeros =
    foreign "atg_zeros"
    (ptr int @-> int @-> int @-> returning t)

  let zeros_like1 =
    foreign "atg_zeros_like1"
    (t @-> returning t)

  let zeros_like2 =
    foreign "atg_zeros_like2"
    (t @-> int @-> returning t)

  let zeros_out =
    foreign "atg_zeros_out"
    (t @-> ptr int @-> int @-> returning t)

end
