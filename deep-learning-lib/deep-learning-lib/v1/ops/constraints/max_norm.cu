/* Copyright 2016, 2019 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Tools/CUDA_Configuration.cuh>
#include "deep-learning-lib/ops/reduce.cuh"
#include "deep-learning-lib/ops/multiply.cuh"
#include "deep-learning-lib/v1/learner/model.cuh"

__global__ void
kernel__cuModel__Set__Regularization__Max_Norm_Constraints(
    var const regularization__max_norm_constraints_received,
    class cuModel *const ptr_cuModel_received) {
  ptr_cuModel_received->Set__Regularization__Max_Norm_Constraints(
      regularization__max_norm_constraints_received);
}

__host__ __device__ bool
cuModel::Set__Regularization__Max_Norm_Constraints(
    var const regularization__max_norm_constraints_received) {
#ifndef __CUDA_ARCH__
  kernel__cuModel__Set__Regularization__Max_Norm_Constraints<<<1,
                                                                       1u>>>(
      regularization__max_norm_constraints_received, this);

  CUDA__Check_Error();
#else
  if (this->regularization__max_norm_constraints !=
      regularization__max_norm_constraints_received) {
    var const tmp_regularization__max_norm_constraints(
        this->regularization__max_norm_constraints);

    this->regularization__max_norm_constraints =
        regularization__max_norm_constraints_received;

    if (tmp_regularization__max_norm_constraints == 0_r &&
        regularization__max_norm_constraints_received != 0_r) {
      if (this->Allocate__Neurons_Reduce_Norms() == false) {
        ERR(
            L"Can not allocate regularization connections!",);

        return false;
      }
    } else if (tmp_regularization__max_norm_constraints != 0_r &&
               regularization__max_norm_constraints_received == 0_r) {
      this->Deallocate__Neurons_Reduce_Norms();
    }
  }
#endif

  return true;
}

__device__ void
cuModel::Update_Weight_Regularization__Max_Norm_Constraints(void) {
  struct cuLayer const *const last_layer(this->ptr_last_layer - 1);
  struct cuLayer const *layer_it(this->ptr_array_layers + 1);

  this->Update_Weight_Regularization__Max_Norm_Constraints__Neurons(
      layer_it, last_layer);
}

template <typename T>
__device__ inline void Vector__Max_Norm_Constraints_Reduce(
    bool &ref_synchronized_received, size_t const number_connections_received,
    size_t const increment_step_dim3_received,
    T *const ptr_array_reduce_norms_received,
    T *const ptr_array_parameters_received,
    struct dim3 const *const ptr_array_dim3_grid_norms_received,
    struct dim3 const *const ptr_array_dim3_block_norms_received) {
  Reduce::Reduce_Square<T>(
      number_connections_received, increment_step_dim3_received,
      ptr_array_reduce_norms_received, ptr_array_parameters_received,
      ptr_array_dim3_grid_norms_received, ptr_array_dim3_block_norms_received);

  // Do we need to synchronise? Based on "Reduce_Square" Function.
  // => Synchronize if needed to see the norms reduced of the neuron.
  if (number_connections_received >= warpSize) {
    // We need a synchronisation here.
    ref_synchronized_received = false;
  }
}

template <typename T>
__device__ inline void Vector__Max_Norm_Constraints_Normalize(
    bool &ref_synchronized_received, size_t const number_connections_received,
    T const regularization__max_norm_constraints_received,
    T *const ptr_array_reduce_norms_received,
    T *const ptr_array_parameters_received,
    struct dim3 const *const ptr_dim3_grid_connections_received,
    struct dim3 const *const ptr_dim3_block_connections_received) {
  T tmp_desired, tmp_desired_max_norm;

  *ptr_array_reduce_norms_received = sqrt(*ptr_array_reduce_norms_received);

  tmp_desired =
      DL::Math::clip<var>(*ptr_array_reduce_norms_received, T(0),
                           regularization__max_norm_constraints_received);

  if (tmp_desired != *ptr_array_reduce_norms_received) {
    tmp_desired_max_norm = tmp_desired / *ptr_array_reduce_norms_received;

    Multiply::Multiply_X_Y_1D(ref_synchronized_received,
                              number_connections_received, tmp_desired_max_norm,
                              ptr_array_parameters_received,
                              ptr_dim3_grid_connections_received,
                              ptr_dim3_block_connections_received);
  }
}

template <typename T>
__global__ void
kernel__Update_Weight_Regularization__Max_Norm_Constraints__Neurons(
    size_t const *const ptr_array_neuroyed_number_neurons_in_layer_received,
    T const regularization__max_norm_constraints_received,
    T **const ptr_array_2D_reduce_norms_received,
    T *const ptr_array_weigths_received,
    size_t const *const ptr_array_first_index_connection_received,
    size_t const *const ptr_array_neuroyed_number_connections_received,
    struct dim3 **const ptr_array_2D_dim3_grid_norms_received,
    struct dim3 **const ptr_array_2D_dim3_block_norms_received,
    struct dim3 const *const ptr_array_dim3_grid_connections_received,
    struct dim3 const *const ptr_array_dim3_block_connections_received) {
  // By default the synchronized state is set to true.
  bool tmp_synchronized(true);

  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_number_connections(ptr_array_neuroyed_number_connections_received
                                 [tmp_thread_global_index]);

  if (tmp_number_connections != 0u)  // If is not a bias.
  {
    Vector__Max_Norm_Constraints_Reduce<T>(
        tmp_synchronized,
        tmp_number_connections - 1_UZ,  // Subtract bias.
        ptr_array_neuroyed_number_neurons_in_layer_received
            [tmp_thread_global_index],
        ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
        ptr_array_weigths_received +
            ptr_array_first_index_connection_received[tmp_thread_global_index],
        ptr_array_2D_dim3_grid_norms_received[tmp_thread_global_index],
        ptr_array_2D_dim3_block_norms_received[tmp_thread_global_index]);
  }

  // Do we need to synchronise? Based on "CUDA__Device_Synchronise" Function.
  // => Synchronisation before using the reduce norms.
  CUDA__Device_Synchronise(tmp_synchronized,
                           DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::
                               TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK);

  if (tmp_number_connections != 0u)  // If is not a bias.
  {
    Vector__Max_Norm_Constraints_Normalize<T>(
        tmp_synchronized,
        tmp_number_connections - 1_UZ,  // Subtract bias.
        regularization__max_norm_constraints_received,
        ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
        ptr_array_weigths_received +
            ptr_array_first_index_connection_received[tmp_thread_global_index],
        ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
        ptr_array_dim3_block_connections_received + tmp_thread_global_index);
  }
}

template <typename T>
__global__ void
kernel__Update_Weight_Regularization__Max_Norm_Constraints__Neurons(
    size_t const size_received,
    size_t const *const ptr_array_neuroyed_number_neurons_in_layer_received,
    T const regularization__max_norm_constraints_received,
    T **const ptr_array_2D_reduce_norms_received,
    T *const ptr_array_weigths_received,
    size_t const *const ptr_array_first_index_connection_received,
    size_t const *const ptr_array_neuroyed_number_connections_received,
    struct dim3 **const ptr_array_2D_dim3_grid_norms_received,
    struct dim3 **const ptr_array_2D_dim3_block_norms_received,
    struct dim3 const *const ptr_array_dim3_grid_connections_received,
    struct dim3 const *const ptr_array_dim3_block_connections_received) {
  // By default the synchronized state is set to true.
  bool tmp_synchronized(true);

  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
  size_t tmp_number_connections;

  if (tmp_thread_global_index < size_received) {
    tmp_number_connections =
        ptr_array_neuroyed_number_connections_received[tmp_thread_global_index];

    if (tmp_number_connections != 0u)  // If is not a bias.
    {
      Vector__Max_Norm_Constraints_Reduce<T>(
          tmp_synchronized,
          tmp_number_connections - 1_UZ,  // Subtract bias.
          ptr_array_neuroyed_number_neurons_in_layer_received
              [tmp_thread_global_index],
          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
          ptr_array_weigths_received + ptr_array_first_index_connection_received
                                           [tmp_thread_global_index],
          ptr_array_2D_dim3_grid_norms_received[tmp_thread_global_index],
          ptr_array_2D_dim3_block_norms_received[tmp_thread_global_index]);
    }
  }

  // Do we need to synchronise? Based on "CUDA__Device_Synchronise" Function.
  // => Synchronisation before using the reduce norms.
  CUDA__Device_Synchronise(tmp_synchronized,
                           DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::
                               TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK);

  if (tmp_thread_global_index < size_received && tmp_number_connections != 0u) {
    // INFO(L"Neuron_unit[%u], nConnections(%u), Norm(%f)",
    //                         tmp_thread_global_index,
    //                         tmp_number_connections,
    //                         *(ptr_array_2D_reduce_norms_received[tmp_thread_global_index]));

    Vector__Max_Norm_Constraints_Normalize<T>(
        tmp_synchronized,
        tmp_number_connections - 1_UZ,  // Subtract bias.
        regularization__max_norm_constraints_received,
        ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
        ptr_array_weigths_received +
            ptr_array_first_index_connection_received[tmp_thread_global_index],
        ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
        ptr_array_dim3_block_connections_received + tmp_thread_global_index);
  }
}

template <typename T>
__global__ void
kernel_while__Update_Weight_Regularization__Max_Norm_Constraints__Neurons(
    size_t const size_received,
    size_t const *const ptr_array_neuroyed_number_neurons_in_layer_received,
    T const regularization__max_norm_constraints_received,
    T **const ptr_array_2D_reduce_norms_received,
    T *const ptr_array_weigths_received,
    size_t const *const ptr_array_first_index_connection_received,
    size_t const *const ptr_array_neuroyed_number_connections_received,
    struct dim3 **const ptr_array_2D_dim3_grid_norms_received,
    struct dim3 **const ptr_array_2D_dim3_block_norms_received,
    struct dim3 const *const ptr_array_dim3_grid_connections_received,
    struct dim3 const *const ptr_array_dim3_block_connections_received) {
  // By default the synchronized state is set to true.
  bool tmp_synchronized(true);

  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_number_connections;

  do {
    tmp_number_connections =
        ptr_array_neuroyed_number_connections_received[tmp_thread_global_index];

    if (tmp_number_connections != 0u)  // If is not a bias.
    {
      Vector__Max_Norm_Constraints_Reduce<T>(
          tmp_synchronized,
          tmp_number_connections - 1_UZ,  // Subtract bias.
          ptr_array_neuroyed_number_neurons_in_layer_received
              [tmp_thread_global_index],
          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
          ptr_array_weigths_received + ptr_array_first_index_connection_received
                                           [tmp_thread_global_index],
          ptr_array_2D_dim3_grid_norms_received[tmp_thread_global_index],
          ptr_array_2D_dim3_block_norms_received[tmp_thread_global_index]);
    }

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);

  // reset index to the initial state.
  tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Do we need to synchronise? Based on "CUDA__Device_Synchronise" Function.
  // => Synchronisation before using the reduce norms.
  CUDA__Device_Synchronise(tmp_synchronized,
                           DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::
                               TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK);

  do {
    tmp_number_connections =
        ptr_array_neuroyed_number_connections_received[tmp_thread_global_index];

    if (tmp_number_connections != 0u)  // If is not a bias.
    {
      Vector__Max_Norm_Constraints_Normalize<T>(
          tmp_synchronized,
          tmp_number_connections - 1_UZ,  // Subtract bias.
          regularization__max_norm_constraints_received,
          ptr_array_2D_reduce_norms_received[tmp_thread_global_index],
          ptr_array_weigths_received + ptr_array_first_index_connection_received
                                           [tmp_thread_global_index],
          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
          ptr_array_dim3_block_connections_received + tmp_thread_global_index);
    }

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

__device__ void cuModel::
    Update_Weight_Regularization__Max_Norm_Constraints__Neurons(
        struct cuLayer const *const layer_it,
        struct cuLayer const *const last_layer) {
  // By default the synchronized state is set to true.
  bool tmp_synchronized(true);

  struct cuNeuron const *const tmp_ptr_last_neuron_unit(
      last_layer->ptr_last_neuron_unit);
  struct cuNeuron *tmp_ptr_neuron_unit_it(
      layer_it->ptr_array_neuron_units);

  size_t const tmp_number_neurons_received(
      static_cast<size_t>(tmp_ptr_last_neuron_unit - tmp_ptr_neuron_unit_it)),
      *tmp_ptr_array_neuroyed_number_neurons_in_layer(
          this->ptr_array_neuroyed_number_neurons_in_layer +
          static_cast<size_t>(tmp_ptr_neuron_unit_it -
                              this->ptr_array_layers->ptr_array_neuron_units));

  if (USE_PARALLEL && tmp_number_neurons_received >= warpSize) {
    // Set the synchronisation state to false. Because we launch a kernel.
    tmp_synchronized = false;

    LAUNCH_KERNEL_1D(
        Update_Weight_Regularization__Max_Norm_Constraints__Neurons<var>,
        this->ptr_array_dim3_grid[6], this->ptr_array_dim3_block[6], 0_UZ,
        tmp_number_neurons_received,
        tmp_ptr_array_neuroyed_number_neurons_in_layer,
        this->regularization__max_norm_constraints,
        tmp_ptr_neuron_unit_it->ptr_array_reduce_norms,
        this->ptr_array_parameters,
        tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index,
        tmp_ptr_neuron_unit_it->ptr_number_forward_connections,
        tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_grid_reduce_norms,
        tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_block_reduce_norms,
        tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections,
        tmp_ptr_neuron_unit_it->ptr_dim3_block_connections)
  } else {
    // Loop through each neuron of the range received as arguments. by default
    // the whole network with connection to it.
    for (; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit;
         ++tmp_ptr_neuron_unit_it,
         ++tmp_ptr_array_neuroyed_number_neurons_in_layer) {
      if (*tmp_ptr_neuron_unit_it->ptr_number_forward_connections !=
          0u)  // If is not a bias.
      {
        Vector__Max_Norm_Constraints_Reduce<var>(
            tmp_synchronized,
            *tmp_ptr_neuron_unit_it->ptr_number_forward_connections -
                1,  // Subtract bias.
            *tmp_ptr_array_neuroyed_number_neurons_in_layer,
            *tmp_ptr_neuron_unit_it->ptr_array_reduce_norms,
            this->ptr_array_parameters +
                *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index,
            *tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_grid_reduce_norms,
            *tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_block_reduce_norms);
      }
    }

    // Synchronize if needed to see the reduced norms of the network.
    CUDA__Device_Synchronise(tmp_synchronized,
                             DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::
                                 TYPE_DEVICE_SYNCHRONIZED_THREAD);

    // Loop through each neuron of the range received as arguments. by default
    // the whole network with connection to it.
    for (tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;
         tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit;
         ++tmp_ptr_neuron_unit_it) {
      if (*tmp_ptr_neuron_unit_it->ptr_number_forward_connections !=
          0u)  // If is not a bias.
      {
        // INFO(L"Neuron_unit[%u], nConnections(%u), Norm(%f)",
        //                         static_cast<size_t>(tmp_ptr_neuron_unit_it -
        //                         layer_it->ptr_array_neuron_units),
        //                         *tmp_ptr_neuron_unit_it->ptr_number_forward_connections,
        //                         *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_norms));

        Vector__Max_Norm_Constraints_Normalize<var>(
            tmp_synchronized,
            *tmp_ptr_neuron_unit_it->ptr_number_forward_connections -
                1,  // Subtract bias.
            this->regularization__max_norm_constraints,
            *tmp_ptr_neuron_unit_it->ptr_array_reduce_norms,
            this->ptr_array_parameters +
                *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index,
            tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections,
            tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);
      }
    }
  }

  // Synchronize if needed to see the weights norms of the network.
  CUDA__Device_Synchronise(tmp_synchronized,
                           DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::
                               TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__host__ __device__ var
cuModel::Get__Regularization__Max_Norm_Constraints(void) const {
  return (this->regularization__max_norm_constraints);
}
