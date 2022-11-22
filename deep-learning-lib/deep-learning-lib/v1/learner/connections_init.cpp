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

#include "deep-learning-lib/pch.hpp"

#include "deep-learning-lib/v1/learner/model.hpp"
#include "deep-learning-lib/v1/mem/reallocate.hpp"

namespace DL::v1 {
template<class U> void Model::Initialize_Connections__FC(Layer *const layer_it, U *const ptr_previous_layer_array_units_received)
{
    size_t tmp_number_connections,
              tmp_connection_index;
    
    void **tmp_ptr_array_ptr_connections;

    var *tmp_ptr_array_parameters;

    Neuron_unit const *const tmp_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit);
    Neuron_unit *tmp_ptr_neuron_unit_it(layer_it->ptr_array_neuron_units);

    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_array_ptr_connections = this->ptr_array_ptr_connections + *tmp_ptr_neuron_unit_it->ptr_first_connection_index;

        tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_connection_index;

        tmp_number_connections = *tmp_ptr_neuron_unit_it->ptr_number_connections;

        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_ptr_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            tmp_ptr_array_parameters[tmp_connection_index] = this->real_gen();
        }
    }
}

template<class U> void Model::Initialize_Connections__LSTM(Layer *const layer_it, U *const ptr_previous_layer_array_units_received)
{
    void **tmp_ptr_array_ptr_cell_input_connections,
          **tmp_ptr_array_ptr_input_gate_connections,
          **tmp_ptr_array_ptr_forget_gate_connections,
          **tmp_ptr_array_ptr_output_gate_connections;
    
    var *tmp_ptr_array_cell_input_parameters,
         *tmp_ptr_array_input_gate_parameters,
         *tmp_ptr_array_forget_gate_parameters,
         *tmp_ptr_array_output_gate_parameters;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);
    
    size_t const tmp_number_peephole_connections(tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate),
                       tmp_number_inputs_connections(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;

    CellUnit const *tmp_ptr_block_ptr_last_cell_unit,
                                    *tmp_ptr_block_ptr_cell_unit_it;
    CellUnit *const tmp_ptr_layer_ptr_first_cell_unit(layer_it->ptr_array_cell_units),
                           *tmp_ptr_block_ptr_first_cell_unit;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        tmp_ptr_block_ptr_first_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

        // [0] Cell input.
        for(tmp_ptr_block_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_block_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
        {
            //    [1] Input, cell input.
            tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

            tmp_ptr_array_ptr_cell_input_connections = this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

            for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
            {
                tmp_ptr_array_cell_input_parameters[tmp_connection_index] = this->real_gen();
                tmp_ptr_array_ptr_cell_input_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            }
            //    [1] |END| Input, cell input. |END|

            //    [1] Recurrent, input.
            tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
            
            tmp_ptr_array_ptr_cell_input_connections = this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

            for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
            {
                tmp_ptr_array_cell_input_parameters[tmp_connection_index] = this->real_gen();
                tmp_ptr_array_ptr_cell_input_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
            }
            //    [1] |END| Recurrent, input. |END|
        }
        // [0] |END| Cell input. |END|
        
        // [0] Input, gates.
        tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
        tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
        tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
        
        tmp_ptr_array_ptr_input_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
        tmp_ptr_array_ptr_forget_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
        tmp_ptr_array_ptr_output_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_parameters[tmp_connection_index] = this->real_gen();
            tmp_ptr_array_forget_gate_parameters[tmp_connection_index] = this->real_gen();
            tmp_ptr_array_output_gate_parameters[tmp_connection_index] = this->real_gen();

            tmp_ptr_array_ptr_input_gate_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            tmp_ptr_array_ptr_forget_gate_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
            tmp_ptr_array_ptr_output_gate_connections[tmp_connection_index] = ptr_previous_layer_array_units_received + tmp_connection_index;
        }
        // [0] |END| Input, gates. |END|

        // [0] Recurrent, gates.
        tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
        tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
        tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
        
        tmp_ptr_array_ptr_input_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
        tmp_ptr_array_ptr_forget_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
        tmp_ptr_array_ptr_output_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_parameters[tmp_connection_index] = this->real_gen();
            tmp_ptr_array_forget_gate_parameters[tmp_connection_index] = this->real_gen();
            tmp_ptr_array_output_gate_parameters[tmp_connection_index] = this->real_gen();

            tmp_ptr_array_ptr_input_gate_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_forget_gate_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_output_gate_connections[tmp_connection_index] = tmp_ptr_layer_ptr_first_cell_unit + tmp_connection_index;
        }
        // [0] |END| Recurrent, gates. |END|

    #ifndef NO_PEEPHOLE
        // [0] Peepholes.
        tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
        tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;
        tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;
        
        tmp_ptr_array_ptr_input_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
        tmp_ptr_array_ptr_forget_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;
        tmp_ptr_array_ptr_output_gate_connections = this->ptr_array_ptr_connections + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_parameters[tmp_connection_index] = this->real_gen();
            tmp_ptr_array_forget_gate_parameters[tmp_connection_index] = this->real_gen();
            tmp_ptr_array_output_gate_parameters[tmp_connection_index] = this->real_gen();

            tmp_ptr_array_ptr_input_gate_connections[tmp_connection_index] = tmp_ptr_block_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_forget_gate_connections[tmp_connection_index] = tmp_ptr_block_ptr_first_cell_unit + tmp_connection_index;
            tmp_ptr_array_ptr_output_gate_connections[tmp_connection_index] = tmp_ptr_block_ptr_first_cell_unit + tmp_connection_index;
        }
        // [0] |END| Peepholes. |END|
    #endif
    }
}

void Model::Initialize_Connections__AF_Ind_Recurrent(Layer *const layer_it)
{
    AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(layer_it->ptr_last_AF_Ind_recurrent_unit);
    AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(layer_it->ptr_array_AF_Ind_recurrent_units);

    for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it)
    {
        this->ptr_array_ptr_connections[*tmp_ptr_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index] = tmp_ptr_AF_Ind_recurrent_unit_it;
        this->ptr_array_parameters[*tmp_ptr_AF_Ind_recurrent_unit_it->ptr_recurrent_connection_index] = this->real_gen();
    }
}

void Model::Initialize_Connections__Bias(Layer *const layer_it)
{
    size_t const tmp_number_connections(layer_it->last_bias_connection_index - layer_it->first_bias_connection_index);

    if(tmp_number_connections != 0_UZ)
    {
        void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + layer_it->first_bias_connection_index);
        Mem::fill_null(tmp_ptr_array_ptr_connections, tmp_ptr_array_ptr_connections + tmp_number_connections);

        var *tmp_ptr_array_parameters(this->ptr_array_parameters + layer_it->first_bias_connection_index);
        VARZERO(tmp_ptr_array_parameters,
                    tmp_number_connections * sizeof(var));
    }
}

void Model::Initialize_Connections__LSTM__Bias(Layer *const layer_it)
{
    size_t const tmp_number_connections(layer_it->last_bias_connection_index - layer_it->first_bias_connection_index);
    
    if(tmp_number_connections != 0_UZ)
    {
        void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + layer_it->first_bias_connection_index);
        Mem::fill_null(tmp_ptr_array_ptr_connections, tmp_ptr_array_ptr_connections + tmp_number_connections);

        // Bias.
        size_t const tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units)),
                           tmp_number_block_units(static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units));

        var *tmp_ptr_array_parameters(this->ptr_array_parameters + layer_it->first_bias_connection_index);

        //  Cell input && Input gate.
        VARZERO(tmp_ptr_array_parameters,
                    tmp_number_cell_units + tmp_number_block_units * sizeof(var));
        tmp_ptr_array_parameters += tmp_number_cell_units + tmp_number_block_units;
        //  |END| Cell input && Input gate. |END|
        
        //  Forget gate.
        for(var const *const tmp_parameter_end(tmp_ptr_array_parameters + tmp_number_block_units); tmp_ptr_array_parameters != tmp_parameter_end; ++tmp_ptr_array_parameters) { *tmp_ptr_array_parameters = 1_r; }
        //  |END| Forget gate. |END|

        //  Output gate.
        VARZERO(tmp_ptr_array_parameters,
                    tmp_number_block_units * sizeof(var));
        //  |END| Output gate. |END|
        // |END| Bias. |END|
    }
}

void Model::Initialize_Connections__FC_to_FC(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<Neuron_unit>(layer_it, ptr_previous_layer_it_received->ptr_array_neuron_units); }

void Model::Initialize_Connections__FC_to_LSTM(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received)
{
    if(layer_it->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<Neuron_unit>(&layer_it->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_neuron_units);
        this->Initialize_Connections__LSTM<Neuron_unit>(&layer_it->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_neuron_units);
    }
    else { this->Initialize_Connections__LSTM<Neuron_unit>(layer_it, ptr_previous_layer_it_received->ptr_array_neuron_units); }
}

void Model::Initialize_Connections__LSTM_to_FC(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<CellUnit>(layer_it, ptr_previous_layer_it_received->ptr_array_cell_units); }

void Model::Initialize_Connections__LSTM_to_LSTM(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received)
{
    if(layer_it->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<CellUnit>(&layer_it->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_cell_units);
        this->Initialize_Connections__LSTM<CellUnit>(&layer_it->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_cell_units);
    }
    else { this->Initialize_Connections__LSTM<CellUnit>(layer_it, ptr_previous_layer_it_received->ptr_array_cell_units); }
}

void Model::Initialize_Connections__Basic_unit_to_FC(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<Basic_unit>(layer_it, ptr_previous_layer_it_received->ptr_array_basic_units); }

void Model::Initialize_Connections__Basic_unit_to_LSTM(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received)
{
    if(layer_it->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<Basic_unit>(&layer_it->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_basic_units);
        this->Initialize_Connections__LSTM<Basic_unit>(&layer_it->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_basic_units);
    }
    else { this->Initialize_Connections__LSTM<Basic_unit>(layer_it, ptr_previous_layer_it_received->ptr_array_basic_units); }
}

void Model::Initialize_Connections__Basic_indice_unit_to_FC(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received) { this->Initialize_Connections__FC<Basic_indice_unit>(layer_it, ptr_previous_layer_it_received->ptr_array_basic_indice_units); }

void Model::Initialize_Connections__Basic_indice_unit_to_LSTM(Layer *const layer_it, Layer const *const ptr_previous_layer_it_received)
{
    if(layer_it->Use__Bidirectional())
    {
        this->Initialize_Connections__LSTM<Basic_indice_unit>(&layer_it->ptr_Bidirectional_Layer->forward_layer, ptr_previous_layer_it_received->ptr_array_basic_indice_units);
        this->Initialize_Connections__LSTM<Basic_indice_unit>(&layer_it->ptr_Bidirectional_Layer->backward_layer, ptr_previous_layer_it_received->ptr_array_basic_indice_units);
    }
    else { this->Initialize_Connections__LSTM<Basic_indice_unit>(layer_it, ptr_previous_layer_it_received->ptr_array_basic_indice_units); }
}
}
