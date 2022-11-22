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

#include "deep-learning-lib/v1/learner/model.cuh"

template<typename T>
__global__ void kernel__cuModel__Assign_Inputs__Dropout_Bernoulli__Training(bool const *const ptr_array_mask_dropout_received,
                                                                                                                 T *const ptr_array_input_layer_value_received,
                                                                                                                 T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(ptr_array_mask_dropout_received[tmp_thread_global_index])
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
    else
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = T(0); }
}

template<typename T>
__global__ void kernel__cuModel__Assign_Inputs__Dropout_Bernoulli__Training(size_t const size_received,
                                                                                                                 bool const *const ptr_array_mask_dropout_received,
                                                                                                                 T *const ptr_array_input_layer_value_received,
                                                                                                                 T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received && ptr_array_mask_dropout_received[tmp_thread_global_index])
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
    else
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = T(0); }
}

template<typename T>
__global__ void kernel_while__cuModel__Assign_Inputs__Dropout_Bernoulli__Training(size_t const size_received,
                                                                                                                          bool const *const ptr_array_mask_dropout_received,
                                                                                                                          T *const ptr_array_input_layer_value_received,
                                                                                                                          T const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(ptr_array_mask_dropout_received[tmp_thread_global_index])
        { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
        else
        { ptr_array_input_layer_value_received[tmp_thread_global_index] = T(0); }

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__cuModel__Assign_Inputs__Dropout_Bernoulli__Testing(T const dropout_values,
                                                                                                                                T *const ptr_array_input_layer_value_received,
                                                                                                                                T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index] * dropout_values;
}

template<typename T>
__global__ void kernel__cuModel__Assign_Inputs__Dropout_Bernoulli__Testing(size_t const size_received,
                                                                                                                                T const dropout_values,
                                                                                                                                T *const ptr_array_input_layer_value_received,
                                                                                                                                T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received)
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index] * dropout_values; }
}

template<typename T>
__global__ void kernel_while__cuModel__Assign_Inputs__Dropout_Bernoulli__Testing(size_t const size_received,
                                                                                                                         T const dropout_values,
                                                                                                                         T *const ptr_array_input_layer_value_received,
                                                                                                                         T const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index] * dropout_values;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__cuModel__Assign_Inputs(T *const ptr_array_input_layer_value_received, T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index];
}

template<typename T>
__global__ void kernel__cuModel__Assign_Inputs(size_t const size_received,
                                                                                     T *const ptr_array_input_layer_value_received,
                                                                                     T const *const ptr_array_inputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    if(tmp_thread_global_index < size_received)
    { ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index]; }
}

template<typename T>
__global__ void kernel_while__cuModel__Assign_Inputs(size_t const size_received,
                                                                                              T *const ptr_array_input_layer_value_received,
                                                                                              T const *const ptr_array_inputs_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
        
    do
    {
        ptr_array_input_layer_value_received[tmp_thread_global_index] = ptr_array_inputs_received[tmp_thread_global_index];

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Assign_Inputs(bool &ref_synchronized_received,
                                                                                 size_t const thread_index_received,
                                                                                 var const *ptr_array_inputs_received)
{
    struct cuLayer const *const tmp_ptr_input_layer(this->ptr_array_layers);
    
    bool const *tmp_ptr_array_input_layer_mask_dropout;

    var *tmp_ptr_array_input_layer_values(tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values + thread_index_received * *tmp_ptr_input_layer->ptr_number_neurons),
         tmp_probability_retained_unit;
    var const *const tmp_ptr_array_input_layers_values_end(tmp_ptr_array_input_layer_values + this->n_inp);

    if(this->use_Dropout)
    {
        if(this->type_state_propagation == DL::PROPAGATION::TRAINING)
        {
            tmp_ptr_array_input_layer_mask_dropout = tmp_ptr_input_layer->ptr_array_neuron_units->ptr_mask_dropout_bernoulli;

            // Condition to enter into dynamic parallelisme of each.
            if(USE_PARALLEL && this->n_inp >= warpSize)
            {
                // Set the synchronisation state to false. Because we launch a kernel.
                ref_synchronized_received = false;
                
                LAUNCH_KERNEL_POINTER_1D(cuModel__Assign_Inputs__Dropout_Bernoulli__Training<var>,
                                                                  tmp_ptr_input_layer->ptr_dim3_grid_neurons,
                                                                  tmp_ptr_input_layer->ptr_dim3_block_neurons,
                                                                  0_UZ,
                                                                  this->n_inp,
                                                                  tmp_ptr_array_input_layer_mask_dropout,
                                                                  tmp_ptr_array_input_layer_values,
                                                                  ptr_array_inputs_received)
            }
            // Standard assignment inputs.
            else
            {
                for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                      ++tmp_ptr_array_input_layer_mask_dropout,
                                                                                                                                      ++ptr_array_inputs_received)
                {
                    if(*tmp_ptr_array_input_layer_mask_dropout)
                    { *tmp_ptr_array_input_layer_values = *ptr_array_inputs_received; }
                    else
                    { *tmp_ptr_array_input_layer_values = 0_r; }
                }
            }
        }
        else
        {
            // Condition to enter into dynamic parallelisme of each.
            if(USE_PARALLEL && this->n_inp >= warpSize)
            {
                // Set the synchronisation state to false. Because we launch a kernel.
                ref_synchronized_received = false;
                
                LAUNCH_KERNEL_POINTER_1D(cuModel__Assign_Inputs__Dropout_Bernoulli__Testing<var>,
                                                                  tmp_ptr_input_layer->ptr_dim3_grid_neurons,
                                                                  tmp_ptr_input_layer->ptr_dim3_block_neurons,
                                                                  0_UZ,
                                                                  this->n_inp,
                                                                  tmp_ptr_input_layer->dropout_values[0],
                                                                  tmp_ptr_array_input_layer_values,
                                                                  ptr_array_inputs_received)
            }
            // Standard assignment inputs.
            else
            {
                tmp_probability_retained_unit = tmp_ptr_input_layer->dropout_values[0];
                
                for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                      ++ptr_array_inputs_received)
                { *tmp_ptr_array_input_layer_values = *ptr_array_inputs_received * tmp_probability_retained_unit; }
            }
        }
    }
    else
    {
        // Condition to enter into dynamic parallelisme of each.
        if(USE_PARALLEL && this->n_inp >= warpSize)
        {
            // Set the synchronisation state to false. Because we launch a kernel.
            ref_synchronized_received = false;
            
            LAUNCH_KERNEL_POINTER_1D(cuModel__Assign_Inputs<var>,
                                                              tmp_ptr_input_layer->ptr_dim3_grid_neurons,
                                                              tmp_ptr_input_layer->ptr_dim3_block_neurons,
                                                              0_UZ,
                                                              this->n_inp,
                                                              tmp_ptr_array_input_layer_values,
                                                              ptr_array_inputs_received)
        }
        // Standard assignment inputs.
        else
        {
            for(; tmp_ptr_array_input_layer_values != tmp_ptr_array_input_layers_values_end; ++tmp_ptr_array_input_layer_values,
                                                                                                                                    ++ptr_array_inputs_received)
            { *tmp_ptr_array_input_layer_values = *ptr_array_inputs_received; }
        }
    }
}

template <typename T>
__global__ void kernel__cuModel__Assign_Input_Batch(
    size_t const size_inputs_received, size_t const number_neurons_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_data_index(tmp_thread_global_index / size_inputs_received),
      tmp_input_index(tmp_thread_global_index % size_inputs_received);

  ptr_array_input_layer_value_received[tmp_data_index *
                                           number_neurons_received +
                                       tmp_input_index] =
      ptr_array_inputs_received[tmp_data_index][tmp_input_index];
}

template <typename T>
__global__ void kernel__cuModel__Assign_Input_Batch(
    size_t const size_received, size_t const size_inputs_received,
    size_t const number_neurons_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    size_t const tmp_data_index(
        (tmp_thread_global_index / size_inputs_received)),
        tmp_input_index(tmp_thread_global_index % size_inputs_received);

    ptr_array_input_layer_value_received[tmp_data_index *
                                             number_neurons_received +
                                         tmp_input_index] =
        ptr_array_inputs_received[tmp_data_index][tmp_input_index];
  }
}

template <typename T>
__global__ void kernel_while__cuModel__Assign_Input_Batch(
    size_t const size_received, size_t const size_inputs_received,
    size_t const number_neurons_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_data_index, tmp_input_index;

  do {
    tmp_data_index = (tmp_thread_global_index / size_inputs_received);
    tmp_input_index = tmp_thread_global_index % size_inputs_received;

    ptr_array_input_layer_value_received[tmp_data_index *
                                             number_neurons_received +
                                         tmp_input_index] =
        ptr_array_inputs_received[tmp_data_index][tmp_input_index];

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__global__ void kernel__cuModel__Assign_Input_Batch__Dropout_Bernoulli__Testing(
    size_t const size_inputs_received, size_t const number_neurons_received,
    T const probability_retained_unit_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_data_index((tmp_thread_global_index / size_inputs_received)),
      tmp_input_index(tmp_thread_global_index % size_inputs_received);

  ptr_array_input_layer_value_received[tmp_data_index *
                                           number_neurons_received +
                                       tmp_input_index] =
      ptr_array_inputs_received[tmp_data_index][tmp_input_index] *
      probability_retained_unit_received;
}

template <typename T>
__global__ void kernel__cuModel__Assign_Input_Batch__Dropout_Bernoulli__Testing(
    size_t const size_received, size_t const size_inputs_received,
    size_t const number_neurons_received,
    T const probability_retained_unit_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    size_t const tmp_data_index(
        (tmp_thread_global_index / size_inputs_received)),
        tmp_input_index(tmp_thread_global_index % size_inputs_received);

    ptr_array_input_layer_value_received[tmp_data_index *
                                             number_neurons_received +
                                         tmp_input_index] =
        ptr_array_inputs_received[tmp_data_index][tmp_input_index] *
        probability_retained_unit_received;
  }
}

template <typename T>
__global__ void
kernel_while__cuModel__Assign_Input_Batch__Dropout_Bernoulli__Testing(
    size_t const size_received, size_t const size_inputs_received,
    size_t const number_neurons_received,
    T const probability_retained_unit_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_data_index, tmp_input_index;

  do {
    tmp_data_index = (tmp_thread_global_index / size_inputs_received);
    tmp_input_index = tmp_thread_global_index % size_inputs_received;

    ptr_array_input_layer_value_received[tmp_data_index *
                                             number_neurons_received +
                                         tmp_input_index] =
        ptr_array_inputs_received[tmp_data_index][tmp_input_index] *
        probability_retained_unit_received;

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__global__ void
kernel__cuModel__Assign_Input_Batch__Dropout_Bernoulli__Training(
    size_t const size_inputs_received, size_t const number_neurons_received,
    bool const *const ptr_array_input_layer_mask_dropout_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_data_index((tmp_thread_global_index / size_inputs_received)),
      tmp_input_index(tmp_thread_global_index % size_inputs_received);

  if (ptr_array_input_layer_mask_dropout_received[tmp_input_index]) {
    ptr_array_input_layer_value_received[tmp_data_index *
                                             number_neurons_received +
                                         tmp_input_index] =
        ptr_array_inputs_received[tmp_data_index][tmp_input_index];
  } else {
    ptr_array_input_layer_value_received[tmp_data_index *
                                             number_neurons_received +
                                         tmp_input_index] = T(0);
  }
}

template <typename T>
__global__ void
kernel__cuModel__Assign_Input_Batch__Dropout_Bernoulli__Training(
    size_t const size_received, size_t const size_inputs_received,
    size_t const number_neurons_received,
    bool const *const ptr_array_input_layer_mask_dropout_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    size_t const tmp_data_index(
        (tmp_thread_global_index / size_inputs_received)),
        tmp_input_index(tmp_thread_global_index % size_inputs_received);

    if (ptr_array_input_layer_mask_dropout_received[tmp_input_index]) {
      ptr_array_input_layer_value_received[tmp_data_index *
                                               number_neurons_received +
                                           tmp_input_index] =
          ptr_array_inputs_received[tmp_data_index][tmp_input_index];
    } else {
      ptr_array_input_layer_value_received[tmp_data_index *
                                               number_neurons_received +
                                           tmp_input_index] = T(0);
    }
  }
}

template <typename T>
__global__ void
kernel_while__cuModel__Assign_Input_Batch__Dropout_Bernoulli__Training(
    size_t const size_received, size_t const size_inputs_received,
    size_t const number_neurons_received,
    bool const *const ptr_array_input_layer_mask_dropout_received,
    T *const ptr_array_input_layer_value_received,
    T const *const *const ptr_array_inputs_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_data_index, tmp_input_index;

  do {
    tmp_data_index = (tmp_thread_global_index / size_inputs_received);
    tmp_input_index = tmp_thread_global_index % size_inputs_received;

    if (ptr_array_input_layer_mask_dropout_received[tmp_input_index]) {
      ptr_array_input_layer_value_received[tmp_data_index *
                                               number_neurons_received +
                                           tmp_input_index] =
          ptr_array_inputs_received[tmp_data_index][tmp_input_index];
    } else {
      ptr_array_input_layer_value_received[tmp_data_index *
                                               number_neurons_received +
                                           tmp_input_index] = T(0);
    }

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

__device__ void cuModel::Assign_Inputs_Batch(bool &ref_synchronized_received,
                                             size_t const batch_size,
                                             var const *const *const Xm) {
  size_t const tmp_batch_size_times_number_inputs(batch_size * this->n_inp);
  size_t tmp_data_index;

  var const *tmp_ptr_array_inputs;

  // Variable to cache optimal size to launch dynamic parallelisme through the
  // GPU.
  struct dim3 tmp_dim3_grid, tmp_dim3_block;

  struct cuLayer *const tmp_ptr_input_layer(this->ptr_array_layers);

  bool const *tmp_ptr_array_input_layer_mask_dropout;

  var const *tmp_ptr_array_input_layers_values_end;
  var *tmp_ptr_array_input_layer_values, tmp_probability_retained_unit;

  // Condition to enter into dynamic parallelisme.
  if (USE_PARALLEL && tmp_batch_size_times_number_inputs >= warpSize) {
    // Set the synchronisation state to false. Because we launch a kernel.
    ref_synchronized_received = false;

    // Get or compute the optimal size to launch dynamic parallelisme through
    // the GPU.
    tmp_ptr_input_layer->ptr_Class_Storage_Dim3_Batch->Get__Dim3_1D(
        tmp_batch_size_times_number_inputs, tmp_dim3_grid, tmp_dim3_block,
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    // Condition to know if we use dropout. For droped inputs.
    if (this->use_Dropout) {
      if (this->type_state_propagation == DL::PROPAGATION::TRAINING) {
        LAUNCH_KERNEL_1D(
            cuModel__Assign_Input_Batch__Dropout_Bernoulli__Training<var>,
            tmp_dim3_grid, tmp_dim3_block, 0_UZ,
            tmp_batch_size_times_number_inputs, this->n_inp,
            *tmp_ptr_input_layer->ptr_number_neurons,
            tmp_ptr_input_layer->ptr_array_neuron_units
                ->ptr_mask_dropout_bernoulli,
            tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values, Xm)
      } else {
        LAUNCH_KERNEL_1D(
            cuModel__Assign_Input_Batch__Dropout_Bernoulli__Testing<var>,
            tmp_dim3_grid, tmp_dim3_block, 0_UZ,
            tmp_batch_size_times_number_inputs, this->n_inp,
            *tmp_ptr_input_layer->ptr_number_neurons,
            tmp_ptr_input_layer->dropout_values[0],
            tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values, Xm)
      }
    }
    // Standard assignment inputs.
    else {
      LAUNCH_KERNEL_1D(
          cuModel__Assign_Input_Batch<var>, tmp_dim3_grid, tmp_dim3_block, 0_UZ,
          tmp_batch_size_times_number_inputs, this->n_inp,
          *tmp_ptr_input_layer->ptr_number_neurons,
          tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values, Xm)
    }
  }
  // If we don't enter into dynamic parallelisme, we serialize the computation.
  else {
    // Condition to know if we use dropout. For droped inputs.
    if (this->use_Dropout) {
      if (this->type_state_propagation == DL::PROPAGATION::TRAINING) {
        // Loop through each sample data.
        for (tmp_data_index = 0_UZ; tmp_data_index != batch_size;
             ++tmp_data_index) {
          // Get inputs array from sample.
          tmp_ptr_array_inputs = Xm[tmp_data_index];

          // Assign value position.
          tmp_ptr_array_input_layer_values =
              tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values +
              tmp_data_index * *tmp_ptr_input_layer->ptr_number_neurons;

          // Assign value end pointer.
          tmp_ptr_array_input_layers_values_end =
              tmp_ptr_array_input_layer_values + this->n_inp;

          // Assign mask dropout.
          tmp_ptr_array_input_layer_mask_dropout =
              tmp_ptr_input_layer->ptr_array_neuron_units
                  ->ptr_mask_dropout_bernoulli;

          // Loop through each input.
          for (; tmp_ptr_array_input_layer_values !=
                 tmp_ptr_array_input_layers_values_end;
               ++tmp_ptr_array_input_layer_values,
               ++tmp_ptr_array_input_layer_mask_dropout,
               ++tmp_ptr_array_inputs) {
            // Condition to see if the entry is alive. If yes, assign an input
            // from sample.
            if (*tmp_ptr_array_input_layer_mask_dropout) {
              *tmp_ptr_array_input_layer_values = *tmp_ptr_array_inputs;
            }
            // Entry dead. Give it a zero value.
            else {
              *tmp_ptr_array_input_layer_values = 0_r;
            }
          }
        }
      } else {
        tmp_probability_retained_unit = tmp_ptr_input_layer->dropout_values[0];

        // Loop through each sample data.
        for (tmp_data_index = 0_UZ; tmp_data_index != batch_size;
             ++tmp_data_index) {
          // Get inputs array from sample.
          tmp_ptr_array_inputs = Xm[tmp_data_index];

          // Assign value position.
          tmp_ptr_array_input_layer_values =
              tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values +
              tmp_data_index * *tmp_ptr_input_layer->ptr_number_neurons;

          // Assign value end pointer.
          tmp_ptr_array_input_layers_values_end =
              tmp_ptr_array_input_layer_values + this->n_inp;

          // Assign mask dropout.
          tmp_ptr_array_input_layer_mask_dropout =
              tmp_ptr_input_layer->ptr_array_neuron_units
                  ->ptr_mask_dropout_bernoulli;

          // Loop through each input.
          for (; tmp_ptr_array_input_layer_values !=
                 tmp_ptr_array_input_layers_values_end;
               ++tmp_ptr_array_input_layer_values,
               ++tmp_ptr_array_input_layer_mask_dropout,
               ++tmp_ptr_array_inputs) {
            *tmp_ptr_array_input_layer_values =
                *tmp_ptr_array_inputs * tmp_probability_retained_unit;
          }
        }
      }
    }
    // Standard assignment inputs.
    else {
      // Loop through each sample data.
      for (tmp_data_index = 0_UZ; tmp_data_index != batch_size;
           ++tmp_data_index) {
        // Get inputs array from sample.
        tmp_ptr_array_inputs = Xm[tmp_data_index];

        // Assign value position.
        tmp_ptr_array_input_layer_values =
            tmp_ptr_input_layer->ptr_array_neuron_units->ptr_array_values +
            tmp_data_index * *tmp_ptr_input_layer->ptr_number_neurons;

        // Assign value end pointer.
        tmp_ptr_array_input_layers_values_end =
            tmp_ptr_array_input_layer_values + this->n_inp;

        // Loop through each input.
        for (; tmp_ptr_array_input_layer_values !=
               tmp_ptr_array_input_layers_values_end;
             ++tmp_ptr_array_input_layer_values, ++tmp_ptr_array_inputs) {
          *tmp_ptr_array_input_layer_values = *tmp_ptr_array_inputs;
        }
      }
    }
  }
}