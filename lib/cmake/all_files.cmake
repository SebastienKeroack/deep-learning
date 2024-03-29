﻿if (LINUX)
  set (CXX_OS_FILES
    "deep-learning/data/time/time_nux.cpp"
    "deep-learning/device/system/info/info_nux.cpp"
    "deep-learning/device/system/shutdown_block/shutdown_block_nux.cpp"
    "deep-learning/device/system/shutdown_block/shutdown_block_nux.hpp"
    "deep-learning/io/file/file_nux.cpp"
    "deep-learning/io/form/dialogbox_nux.cpp"
    "deep-learning/io/term/command_nux.cpp"
    "deep-learning/io/term/keyboard_nux.cpp"
    "deep-learning/io/term/keyboard_nux.hpp")
elseif (WIN32)
  set (CXX_OS_FILES
    "deep-learning/data/time/time_win.cpp"
    "deep-learning/device/system/info/info_win.cpp"
    "deep-learning/device/system/shutdown_block/shutdown_block_win.cpp"
    "deep-learning/device/system/shutdown_block/shutdown_block_win.hpp"
    "deep-learning/io/file/file_win.cpp"
    "deep-learning/io/form/dialogbox_win.cpp"
    "deep-learning/io/term/command_win.cpp"
    "deep-learning/io/term/keyboard_win.cpp"
    "deep-learning/io/term/keyboard_win.hpp")
endif ()

set (CXX_CU_FILES
  "deep-learning/v1/data/dataset.cuh"
  "deep-learning/v1/data/datasets.cuh"
  "deep-learning/v1/data/dims.cu"
  "deep-learning/v1/data/dims.cuh"
  "deep-learning/v1/data/shared_memory.cuh"
  "deep-learning/v1/data/dataset/dataset.cu"
  "deep-learning/v1/data/dataset/datasets.cu"
  "deep-learning/v1/device/gpu/cuda/framework.cu"
  "deep-learning/v1/device/gpu/cuda/framework.cuh"
  "deep-learning/v1/device/gpu/cuda/info.cu"
  "deep-learning/v1/device/gpu/cuda/info.cuh"
  "deep-learning/v1/device/gpu/cuda/prop.cu"
  "deep-learning/v1/device/gpu/cuda/prop.cuh"
  "deep-learning/v1/learner/allocate.cu"
  "deep-learning/v1/learner/assign_inputs.cu"
  "deep-learning/v1/learner/bwp.cu"
  "deep-learning/v1/learner/bwp_fc_to_fc.cu"
  "deep-learning/v1/learner/bwp_fc_to_fc_batch_renorm.cu"
  "deep-learning/v1/learner/bwp_fc_to_fc_batch_renorm_dropout.cu"
  "deep-learning/v1/learner/bwp_fc_to_fc_dropout.cu"
  "deep-learning/v1/learner/compute_error.cu"
  "deep-learning/v1/learner/compute_error_bce.cu"
  "deep-learning/v1/learner/compute_error_bit_fail.cu"
  "deep-learning/v1/learner/compute_error_standard.cu"
  "deep-learning/v1/learner/compute_loss.cu"
  "deep-learning/v1/learner/compute_loss_bce.cu"
  "deep-learning/v1/learner/compute_loss_bit_fail.cu"
  "deep-learning/v1/learner/compute_loss_standard.cu"
  "deep-learning/v1/learner/copy.cu"
  "deep-learning/v1/learner/deallocate.cu"
  "deep-learning/v1/learner/dropout.cu"
  "deep-learning/v1/learner/fwp.cu"
  "deep-learning/v1/learner/fwp_fc_to_fc.cu"
  "deep-learning/v1/learner/fwp_fc_to_fc_batch_renorm.cu"
  "deep-learning/v1/learner/fwp_fc_to_fc_batch_renorm_dropout.cu"
  "deep-learning/v1/learner/fwp_fc_to_fc_dropout.cu"
  "deep-learning/v1/learner/model.cu"
  "deep-learning/v1/learner/model.cuh"
  "deep-learning/v1/learner/reallocate.cu"
  "deep-learning/v1/learner/train.cu"
  "deep-learning/v1/learner/update_derivative.cu"
  "deep-learning/v1/learner/update_derivative_fc.cu"
  "deep-learning/v1/learner/update_derivative_fc_dropout.cu"
  "deep-learning/v1/mem/copy.cu"
  "deep-learning/v1/mem/reallocate.cu"
  "deep-learning/v1/mem/reallocate.cuh"
  "deep-learning/v1/mem/reallocate_c.cu"
  "deep-learning/v1/mem/reallocate_c.cuh"
  "deep-learning/v1/ops/accumulate.cu"
  "deep-learning/v1/ops/accumulate.cuh"
  "deep-learning/v1/ops/fill.cu"
  "deep-learning/v1/ops/index.cu"
  "deep-learning/v1/ops/mask.cu"
  "deep-learning/v1/ops/mask.cuh"
  "deep-learning/v1/ops/math.cuh"
  "deep-learning/v1/ops/multiply.cu"
  "deep-learning/v1/ops/multiply.cuh"
  "deep-learning/v1/ops/reduce.cu"
  "deep-learning/v1/ops/reduce.cuh"
  "deep-learning/v1/ops/transpose.cu"
  "deep-learning/v1/ops/transpose.cuh"
  "deep-learning/v1/ops/zero.cu"
  "deep-learning/v1/ops/activations/functions.cuh"
  "deep-learning/v1/ops/constraints/max_norm.cu"
  "deep-learning/v1/ops/distributions/curand.cu"
  "deep-learning/v1/ops/distributions/curand.cuh"
  "deep-learning/v1/ops/distributions/shuffle.cu"
  "deep-learning/v1/ops/distributions/shuffle.cuh"
  "deep-learning/v1/ops/regularizers/l1.cu"
  "deep-learning/v1/ops/regularizers/l2.cu"
  "deep-learning/v1/optimizers/adam.cu"
  "deep-learning/v1/optimizers/amsgrad.cu"
  "deep-learning/v1/optimizers/gradient_descent.cu"
  "deep-learning/v1/optimizers/gradient_descent_momentum.cu"
  "deep-learning/v1/optimizers/irprop_plus.cu"
  "deep-learning/v1/optimizers/nesterov_accelerated_gradient.cu")

set (CXX_FILES
  $<$<CONFIG:Debug>:
    "deep-learning/v1/learner/compute_grad.cpp"
  >
  "deep-learning/session.cpp"
  "deep-learning/session.hpp"
  "deep-learning/data/dataset.hpp"
  "deep-learning/data/dtypes.hpp"
  "deep-learning/data/shape.cpp"
  "deep-learning/data/shape.hpp"
  "deep-learning/data/string.cpp"
  "deep-learning/data/string.hpp"
  "deep-learning/data/time.hpp"
  "deep-learning/data/dataset/dataset.cpp"
  "deep-learning/data/dataset/mnist.cpp"
  "deep-learning/data/dataset/mnist.hpp"
  "deep-learning/data/enum/dialogbox.hpp"
  "deep-learning/data/enum/env.hpp"
  "deep-learning/data/enum/hierarchy.hpp"
  "deep-learning/data/enum/loglevel.hpp"
  "deep-learning/data/time/time.cpp"
  "deep-learning/device/system/info.hpp"
  "deep-learning/device/system/shutdown_block.hpp"
  "deep-learning/device/system/info/info.cpp"
  "deep-learning/drivers/driver.cpp"
  "deep-learning/drivers/driver.hpp"
  "deep-learning/io/file.hpp"
  "deep-learning/io/flags.hpp"
  "deep-learning/io/logger.cpp"
  "deep-learning/io/logger.hpp"
  "deep-learning/io/file/file.cpp"
  "deep-learning/io/form/dialogbox.cpp"
  "deep-learning/io/form/dialogbox.hpp"
  "deep-learning/io/term/command.hpp"
  "deep-learning/io/term/flags.cpp"
  "deep-learning/io/term/input.cpp"
  "deep-learning/io/term/input.hpp"
  "deep-learning/io/term/keyboard.hpp"
  "deep-learning/io/term/spinner.cpp"
  "deep-learning/io/term/spinner.hpp"
  "deep-learning/nn/checkpoint.cpp"
  "deep-learning/nn/checkpoint.hpp"
  "deep-learning/nn/checkpointer.cpp"
  "deep-learning/nn/checkpointer.hpp"
  "deep-learning/nn/learner.cpp"
  "deep-learning/nn/learner.hpp"
  "deep-learning/nn/var.cpp"
  "deep-learning/nn/var.hpp"
  "deep-learning/ops/_math.hpp"
  "deep-learning/ops/math.cpp"
  "deep-learning/ops/math.hpp"
  "deep-learning/ops/modwt.cpp"
  "deep-learning/ops/modwt.hpp"
  "deep-learning/ops/distributions/bernoulli.cpp"
  "deep-learning/ops/distributions/bernoulli.hpp"
  "deep-learning/ops/distributions/distribution.cpp"
  "deep-learning/ops/distributions/distribution.hpp"
  "deep-learning/ops/distributions/gaussian.cpp"
  "deep-learning/ops/distributions/gaussian.hpp"
  "deep-learning/ops/distributions/integer.cpp"
  "deep-learning/ops/distributions/integer.hpp"
  "deep-learning/ops/distributions/real.cpp"
  "deep-learning/ops/distributions/real.hpp"
  "deep-learning/v1/data/datasets.hpp"
  "deep-learning/v1/data/scaler.cpp"
  "deep-learning/v1/data/scaler.hpp"
  "deep-learning/v1/data/dataset/cross_val.cpp"
  "deep-learning/v1/data/dataset/cross_val_opt.cpp"
  "deep-learning/v1/data/dataset/dataset_v1.cpp"
  "deep-learning/v1/data/dataset/datasets.cpp"
  "deep-learning/v1/data/dataset/minibatch.cpp"
  "deep-learning/v1/data/enum/activation.hpp"
  "deep-learning/v1/data/enum/dataset.hpp"
  "deep-learning/v1/data/enum/group.hpp"
  "deep-learning/v1/data/enum/hierarchy.hpp"
  "deep-learning/v1/data/enum/hyperopt.hpp"
  "deep-learning/v1/data/enum/initializer.hpp"
  "deep-learning/v1/data/enum/layer.hpp"
  "deep-learning/v1/data/enum/layer_activation.hpp"
  "deep-learning/v1/data/enum/layer_dropout.hpp"
  "deep-learning/v1/data/enum/layer_norm.hpp"
  "deep-learning/v1/data/enum/loss_fn.hpp"
  "deep-learning/v1/data/enum/model.hpp"
  "deep-learning/v1/data/enum/optimizer.hpp"
  "deep-learning/v1/data/enum/propagation.hpp"
  "deep-learning/v1/data/enum/while.hpp"
  "deep-learning/v1/learner/allocate.cpp"
  "deep-learning/v1/learner/assign_inputs.cpp"
  "deep-learning/v1/learner/batch_normalization.cpp"
  "deep-learning/v1/learner/batch_renormalization.cpp"
  "deep-learning/v1/learner/bernoulli_mp.cpp"
  "deep-learning/v1/learner/bernoulli_st.cpp"
  "deep-learning/v1/learner/bwp_mp.cpp"
  "deep-learning/v1/learner/bwp_rec_mp.cpp"
  "deep-learning/v1/learner/bwp_rec_st.cpp"
  "deep-learning/v1/learner/bwp_st.cpp"
  "deep-learning/v1/learner/compile.cpp"
  "deep-learning/v1/learner/compute_error.cpp"
  "deep-learning/v1/learner/compute_error_mp.cpp"
  "deep-learning/v1/learner/compute_error_st.cpp"
  "deep-learning/v1/learner/compute_loss.cpp"
  "deep-learning/v1/learner/compute_loss_mp.cpp"
  "deep-learning/v1/learner/compute_loss_st.cpp"
  "deep-learning/v1/learner/compute_r.cpp"
  "deep-learning/v1/learner/compute_r_mp.cpp"
  "deep-learning/v1/learner/compute_r_st.cpp"
  "deep-learning/v1/learner/connections_init.cpp"
  "deep-learning/v1/learner/connections_prepare.cpp"
  "deep-learning/v1/learner/copy.cpp"
  "deep-learning/v1/learner/deallocate.cpp"
  "deep-learning/v1/learner/dropout.cpp"
  "deep-learning/v1/learner/fwp_mp.cpp"
  "deep-learning/v1/learner/fwp_rec_mp.cpp"
  "deep-learning/v1/learner/fwp_rec_st.cpp"
  "deep-learning/v1/learner/fwp_st.cpp"
  "deep-learning/v1/learner/indexing.cpp"
  "deep-learning/v1/learner/k_sparse.cpp"
  "deep-learning/v1/learner/load.cpp"
  "deep-learning/v1/learner/loss.cpp"
  "deep-learning/v1/learner/merge.cpp"
  "deep-learning/v1/learner/metrics.cpp"
  "deep-learning/v1/learner/model.cpp"
  "deep-learning/v1/learner/model.hpp"
  "deep-learning/v1/learner/models.cpp"
  "deep-learning/v1/learner/models.hpp"
  "deep-learning/v1/learner/mp.cpp"
  "deep-learning/v1/learner/normalization.cpp"
  "deep-learning/v1/learner/ordering.cpp"
  "deep-learning/v1/learner/reallocate_batch.cpp"
  "deep-learning/v1/learner/reallocate_parameters.cpp"
  "deep-learning/v1/learner/reallocate_threads.cpp"
  "deep-learning/v1/learner/save.cpp"
  "deep-learning/v1/learner/tied_norm.cpp"
  "deep-learning/v1/learner/tied_paramater.cpp"
  "deep-learning/v1/learner/tied_weight.cpp"
  "deep-learning/v1/learner/train.cpp"
  "deep-learning/v1/learner/transfer_learning.cpp"
  "deep-learning/v1/learner/update_derivative_fwp_mp.cpp"
  "deep-learning/v1/learner/update_derivative_fwp_st.cpp"
  "deep-learning/v1/learner/update_derivative_rec_mp.cpp"
  "deep-learning/v1/learner/update_derivative_rec_st.cpp"
  "deep-learning/v1/learner/update_size.cpp"
  "deep-learning/v1/learner/user_controls.cpp"
  "deep-learning/v1/learner/zoneout_mp.cpp"
  "deep-learning/v1/learner/zoneout_st.cpp"
# "deep-learning/v1/mem/reallocate.cpp"
  "deep-learning/v1/mem/reallocate.hpp"
# "deep-learning/v1/mem/reallocate_c.cpp"
  "deep-learning/v1/mem/reallocate_c.hpp"
  "deep-learning/v1/ops/while.hpp"
  "deep-learning/v1/ops/activations/functions.hpp"
  "deep-learning/v1/ops/constraints/clip_gradient.cpp"
  "deep-learning/v1/ops/constraints/constraint.cpp"
  "deep-learning/v1/ops/constraints/euclidean.cpp"
  "deep-learning/v1/ops/constraints/max_norm.cpp"
  "deep-learning/v1/ops/initializers/glorot.cpp"
  "deep-learning/v1/ops/initializers/identity.cpp"
  "deep-learning/v1/ops/initializers/lsuv.cpp"
  "deep-learning/v1/ops/initializers/orthogonal.cpp"
  "deep-learning/v1/ops/initializers/uniform.cpp"
  "deep-learning/v1/ops/regularizers/l1.cpp"
  "deep-learning/v1/ops/regularizers/l2.cpp"
  "deep-learning/v1/ops/regularizers/srip.cpp"
  "deep-learning/v1/ops/regularizers/weight_decay.cpp"
  "deep-learning/v1/optimizers/adabound.cpp"
  "deep-learning/v1/optimizers/adam.cpp"
  "deep-learning/v1/optimizers/amsbound.cpp"
  "deep-learning/v1/optimizers/amsgrad.cpp"
  "deep-learning/v1/optimizers/gaussian_search.cpp"
  "deep-learning/v1/optimizers/gradient_descent.cpp"
  "deep-learning/v1/optimizers/gradient_descent_momentum.cpp"
  "deep-learning/v1/optimizers/grid_search.cpp"
  "deep-learning/v1/optimizers/grid_search.hpp"
  "deep-learning/v1/optimizers/hyperop.cpp"
  "deep-learning/v1/optimizers/irprop_minus.cpp"
  "deep-learning/v1/optimizers/irprop_plus.cpp"
  "deep-learning/v1/optimizers/nesterov_accelerated_gradient.cpp"
  "deep-learning/v1/optimizers/nosadam.cpp"
  "deep-learning/v1/optimizers/quickprop.cpp"
  "deep-learning/v1/optimizers/quickprop.hpp"
  "deep-learning/v1/optimizers/sarprop.cpp"
  "deep-learning/v1/optimizers/sarprop.hpp"
  "framework.hpp"
  "pch.cpp"
  "pch.hpp")

set (ALL_FILES
  ${CXX_FILES}
  ${CXX_OS_FILES})