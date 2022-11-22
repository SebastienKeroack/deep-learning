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
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/io/logger.hpp"
#include "deep-learning-lib/v1/ops/activations/functions.hpp"

namespace DL::v1 {
bool Layer::Use__Coded_Dropout(void) const { return(this->use_coded_dropout); }

bool Model::set_dropout(size_t const index_layer_received,
                                                    LAYER_DROPOUT::TYPE const type_layer_dropout_received,
    real const value_dropout_received[],
                                                    bool const scale_weights_received)
{
    if(index_layer_received >= this->total_layers)
    {
        ERR(L"Layer received (%zu) overflow the number of layers (%zu) in the neural network.",
                                 index_layer_received,
                                 this->total_layers);

        return false;
    }
    else if(this->ptr_array_layers == nullptr)
    {
        ERR(L"\"ptr_array_layers\" is a nullptr.",);

        return false;
    }

    return(this->set_dropout(this->ptr_array_layers + index_layer_received,
                                          type_layer_dropout_received,
                                          value_dropout_received,
                                          scale_weights_received));
}

bool Model::set_dropout(Layer *const ptr_layer_received,
                                                    LAYER_DROPOUT::TYPE const type_layer_dropout_received,
    real const value_dropout_received[],
                                                    bool const scale_weights_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    
    if(this->type == MODEL::AUTOENCODER)
    {
        if(ptr_layer_received == this->ptr_last_layer - (this->total_layers - 3_UZ) / 2_UZ + 2_UZ)
        {
            ERR(L"Layer received as argument is a coded layer.",);

            return false;
        }
        else if(ptr_layer_received >= this->ptr_last_layer - (this->total_layers - 3_UZ) / 2_UZ + 1_UZ)
        {
            ERR(L"Layer received as argument is a decoded layer.",);

            return false;
        }
    }

    switch(type_layer_dropout_received)
    {
        case LAYER_DROPOUT::NONE: return(this->Set__Dropout_None(ptr_layer_received));
        case LAYER_DROPOUT::ALPHA: return(this->Set__Dropout_Alpha(ptr_layer_received, value_dropout_received[0]));
        case LAYER_DROPOUT::BERNOULLI:
            return(this->Set__Dropout_Bernoulli(ptr_layer_received,
                                                                 value_dropout_received[0],
                                                                 scale_weights_received));
        case LAYER_DROPOUT::BERNOULLI_INVERTED: return(this->Set__Dropout_Bernoulli_Inverted(ptr_layer_received, value_dropout_received[0]));
        case LAYER_DROPOUT::GAUSSIAN: return(this->Set__Dropout_Gaussian(ptr_layer_received, value_dropout_received[0]));
        case LAYER_DROPOUT::SHAKEDROP: return(this->Set__Dropout_ShakeDrop(ptr_layer_received, value_dropout_received[0]));
        case LAYER_DROPOUT::UOUT: return(this->Set__Dropout_Uout(ptr_layer_received, value_dropout_received[0]));
        case LAYER_DROPOUT::ZONEOUT:
            return(this->Set__Dropout_Zoneout(ptr_layer_received,
                                                                value_dropout_received[0],
                                                                value_dropout_received[1]));
        default: return false;
    }
}

bool Model::Set__Dropout_None(Layer *const ptr_layer_received)
{
    switch(ptr_layer_received->type_dropout)
    {
        case LAYER_DROPOUT::NONE: return true;
        case LAYER_DROPOUT::ALPHA: return(this->Set__Dropout_Alpha(ptr_layer_received, 0_r));
        case LAYER_DROPOUT::BERNOULLI: return(this->Set__Dropout_Bernoulli(ptr_layer_received, 1_r));
        case LAYER_DROPOUT::BERNOULLI_INVERTED: return(this->Set__Dropout_Bernoulli_Inverted(ptr_layer_received, 1_r));
        case LAYER_DROPOUT::GAUSSIAN: return(this->Set__Dropout_Gaussian(ptr_layer_received, 0_r));
        case LAYER_DROPOUT::SHAKEDROP: return(this->Set__Dropout_ShakeDrop(ptr_layer_received, 0_r));
        case LAYER_DROPOUT::UOUT: return(this->Set__Dropout_Uout(ptr_layer_received, 0_r));
        case LAYER_DROPOUT::ZONEOUT:
            return(this->Set__Dropout_Zoneout(ptr_layer_received,
                                                                0_r,
                                                                0_r));
        default:
            ERR(L"Dropout layer type (%d | %ls) is not managed in the switch.",
                                     ptr_layer_received->type_dropout,
                                     LAYER_DROPOUT_NAME[ptr_layer_received->type_dropout].c_str());
                return false;
    }
}

bool Model::Set__Dropout_Alpha(Layer *const ptr_layer_received, real const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(dropout_probability_received < 0_r)
    {
        ERR(L"probability of dropout (%f) less than zero.",
                                 dropout_probability_received);
        return false;
    }
    else if(dropout_probability_received > 1_r)
    {
        ERR(L"probability of retention (%f) bigger than one.",
                                 dropout_probability_received);

        return false;
    }
    
    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::ALPHA || ptr_layer_received->dropout_values[0] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::ALPHA)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }
        
        real const tmp_keep_probability(1_r - dropout_probability_received);

        ptr_layer_received->dropout_values[0] = tmp_keep_probability;

        if(dropout_probability_received != 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::ALPHA;
            
            real const tmp_alpha(-SELU_Scale * SELU_Alpha);

            ptr_layer_received->dropout_values[1] = pow(tmp_keep_probability + pow(tmp_alpha, 2_r) * tmp_keep_probability * dropout_probability_received, -0.5_r);
            ptr_layer_received->dropout_values[2] = -ptr_layer_received->dropout_values[1] * dropout_probability_received * tmp_alpha;

            if(++this->total_dropout_alpha_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_Bernoulli() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_Bernoulli()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_alpha_layers;

                    return false;
                }
                else if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Neuron__Mask_Dropout_Bernoulli()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_alpha_layers;

                    return false;
                }
            }
        }
        else if(dropout_probability_received == 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::ALPHA)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;
            
            ptr_layer_received->dropout_values[1] = 0_r;
            ptr_layer_received->dropout_values[2] = 0_r;

            if(this->total_dropout_alpha_layers != 0_UZ
              &&
              --this->total_dropout_alpha_layers == 0_UZ
              &&
                    (this->Use__Dropout__Bernoulli() == false
                     &&
                     this->Use__Dropout__Bernoulli__Inverted() == false
                     &&
                     this->Use__Dropout__Alpha() == false))
            {
                this->Deallocate__Generator__Dropout_Bernoulli();

                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return true;
}

bool Model::Set__Dropout_Bernoulli(Layer *const ptr_layer_received,
                                                                  real const retention_probability_received,
                                                                  bool const scale_weights_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(retention_probability_received < 0_r)
    {
        ERR(L"probability of retention (%f) less than zero.",
                                 retention_probability_received);

        return false;
    }
    else if(retention_probability_received > 1_r)
    {
        ERR(L"probability of retention (%f) bigger than one.",
                                 retention_probability_received);

        return false;
    }

    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::BERNOULLI || ptr_layer_received->dropout_values[0] != retention_probability_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::BERNOULLI)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }
        
        if(scale_weights_received
          &&
          ptr_layer_received != this->ptr_array_layers
          &&
          ptr_layer_received->dropout_values[0] != retention_probability_received) { this->Scale_Weight__Dropout(ptr_layer_received->dropout_values[0] / retention_probability_received, ptr_layer_received); }
        
        ptr_layer_received->dropout_values[0] = retention_probability_received;

        if(retention_probability_received != 1_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::BERNOULLI;

            if(++this->total_dropout_bernoulli_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_Bernoulli() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_Bernoulli()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_bernoulli_layers;

                    return false;
                }
                else if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Neuron__Mask_Dropout_Bernoulli()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_bernoulli_layers;

                    return false;
                }
            }
        }
        else if(retention_probability_received == 1_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::BERNOULLI)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;
            
            if(this->total_dropout_bernoulli_layers != 0_UZ
              &&
              --this->total_dropout_bernoulli_layers == 0_UZ
              &&
                    (this->Use__Dropout__Bernoulli() == false
                     &&
                     this->Use__Dropout__Bernoulli__Inverted() == false
                     &&
                     this->Use__Dropout__Alpha() == false))
            {
                this->Deallocate__Generator__Dropout_Bernoulli();

                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return true;
}

bool Model::Set__Dropout_Bernoulli_Inverted(Layer *const ptr_layer_received, real const retention_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(retention_probability_received < 0_r)
    {
        ERR(L"probability of retention (%f) less than zero.",
                                 retention_probability_received);

        return false;
    }
    else if(retention_probability_received > 1_r)
    {
        ERR(L"probability of retention (%f) bigger than one.",
                                 retention_probability_received);

        return false;
    }

    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::BERNOULLI_INVERTED || ptr_layer_received->dropout_values[0] != retention_probability_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::BERNOULLI_INVERTED)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }

        ptr_layer_received->dropout_values[0] = retention_probability_received;

        if(retention_probability_received != 1_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::BERNOULLI_INVERTED;

            if(++this->total_dropout_bernoulli_inverted_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_Bernoulli() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_Bernoulli()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_bernoulli_inverted_layers;

                    return false;
                }
                else if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Neuron__Mask_Dropout_Bernoulli()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_bernoulli_inverted_layers;

                    return false;
                }
            }
        }
        else if(retention_probability_received == 1_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::BERNOULLI_INVERTED)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

            if(this->total_dropout_bernoulli_inverted_layers != 0_UZ
              &&
              --this->total_dropout_bernoulli_inverted_layers == 0_UZ
              &&
                    (this->Use__Dropout__Bernoulli() == false
                     &&
                     this->Use__Dropout__Bernoulli__Inverted() == false
                     &&
                     this->Use__Dropout__Alpha() == false))
            {
                this->Deallocate__Generator__Dropout_Bernoulli();

                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return true;
}

bool Model::Set__Dropout_Gaussian(Layer *const ptr_layer_received, real const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(dropout_probability_received < 0_r)
    {
        ERR(L"probability of dropout (%f) less than zero.",
                                 dropout_probability_received);
        return false;
    }
    else if(dropout_probability_received > 1_r)
    {
        ERR(L"probability of retention (%f) bigger than one.",
                                 dropout_probability_received);

        return false;
    }
    
    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::GAUSSIAN || ptr_layer_received->dropout_values[0] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::GAUSSIAN)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }

        //ptr_layer_received->dropout_values[0] = dropout_probability_received == 1_r ? 0_r : static_cast<real>(pow(sqrt(static_cast<double>(dropout_probability_received) / (1.0 - static_cast<double>(dropout_probability_received))), 2.0));
        ptr_layer_received->dropout_values[0] = dropout_probability_received == 1_r ? 0_r : static_cast<real>(static_cast<double>(dropout_probability_received) / (1.0 - static_cast<double>(dropout_probability_received)));

        if(dropout_probability_received != 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::GAUSSIAN;

            if(++this->total_dropout_gaussian_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_Gaussian() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_Gaussian()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_gaussian_layers;

                    return false;
                }
            }
        }
        else if(dropout_probability_received == 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::GAUSSIAN)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

            if(this->total_dropout_gaussian_layers != 0_UZ && --this->total_dropout_gaussian_layers == 0_UZ) { this->Deallocate__Generator__Dropout_Gaussian(); }
        }
    }

    return true;
}

bool Model::Set__Dropout_ShakeDrop(Layer *const ptr_layer_received, real const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received->type_layer != LAYER::RESIDUAL)
    {
        ERR(L"Layer received as argument is not a residual layer.",);

        return false;
    }
    else if(dropout_probability_received < 0_r)
    {
        ERR(L"probability of dropout (%f) less than zero.",
                                 dropout_probability_received);
        return false;
    }
    else if(dropout_probability_received > 1_r)
    {
        ERR(L"probability of retention (%f) bigger than one.",
                                 dropout_probability_received);

        return false;
    }
    
    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::SHAKEDROP || ptr_layer_received->dropout_values[0] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::SHAKEDROP)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }

        // The paper recommends linear decay rule to determine pl (pl = layer dropout probability, pL = initial dropout probability).
        // l = block index, L = total block.
        // pl = 1 - ( (l / L * (1 - pL) )
        ptr_layer_received->dropout_values[0] = dropout_probability_received;

        if(dropout_probability_received != 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::SHAKEDROP;

            if(++this->total_dropout_shakedrop_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_ShakeDrop() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_ShakeDrop()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_shakedrop_layers;

                    return false;
                }
                else if(this->Allocate__Layer__Mask__Dropout__ShakeDrop() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Layer__Mask__Dropout__ShakeDrop()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_shakedrop_layers;

                    return false;
                }
            }
        }
        else if(dropout_probability_received == 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::SHAKEDROP)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

            if(this->total_dropout_shakedrop_layers != 0_UZ && --this->total_dropout_shakedrop_layers == 0_UZ)
            {
                this->Deallocate__Generator__Dropout_ShakeDrop();
                
                this->Deallocate__Layer__Mask_Dropout_ShakeDrop();
            }
        }
    }

    return true;
}

bool Model::Set__Dropout_Uout(Layer *const ptr_layer_received, real const dropout_probability_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(dropout_probability_received < 0_r)
    {
        ERR(L"probability of dropout (%f) less than zero.",
                                 dropout_probability_received);
        return false;
    }
    else if(dropout_probability_received > 1_r)
    {
        ERR(L"probability of retention (%f) bigger than one.",
                                 dropout_probability_received);

        return false;
    }
    
    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::UOUT || ptr_layer_received->dropout_values[0] != dropout_probability_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::UOUT)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }

        ptr_layer_received->dropout_values[0] = dropout_probability_received;

        if(dropout_probability_received != 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::UOUT;

            if(++this->total_dropout_uout_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_Uout() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_Uout()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_uout_layers;

                    return false;
                }
            }
        }
        else if(dropout_probability_received == 0_r && ptr_layer_received->type_dropout == LAYER_DROPOUT::UOUT)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

            if(this->total_dropout_uout_layers != 0_UZ && --this->total_dropout_uout_layers == 0_UZ) { this->Deallocate__Generator__Dropout_Uout(); }
        }
    }

    return true;
}

bool Model::Set__Dropout_Zoneout(Layer *const ptr_layer_received,
                                                                  real const zoneout_cell_received,
                                                                  real const zoneout_hidden_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"\"ptr_layer_received\" is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(zoneout_cell_received < 0_r)
    {
        ERR(L"probability of zoneout cell (%f) less than zero.",
                                 zoneout_cell_received);
        return false;
    }
    else if(zoneout_cell_received > 1_r)
    {
        ERR(L"probability of zoneout cell (%f) bigger than one.",
                                 zoneout_cell_received);

        return false;
    }
    else if(zoneout_hidden_received < 0_r)
    {
        ERR(L"probability of zoneout hidden (%f) less than zero.",
                                 zoneout_hidden_received);
        return false;
    }
    else if(zoneout_hidden_received > 1_r)
    {
        ERR(L"probability of zoneout hidden (%f) bigger than one.",
                                 zoneout_hidden_received);

        return false;
    }
    
    if(ptr_layer_received->type_dropout != LAYER_DROPOUT::ZONEOUT
       ||
       ptr_layer_received->dropout_values[0] != zoneout_cell_received
       ||
       ptr_layer_received->dropout_values[1] != zoneout_hidden_received)
    {
        if(ptr_layer_received->type_dropout != LAYER_DROPOUT::ZONEOUT)
        {
            if(this->Set__Dropout_None(ptr_layer_received) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Dropout_None(ptr)\" function.",);

                return false;
            }
        }

        ptr_layer_received->dropout_values[0] = zoneout_cell_received;
        ptr_layer_received->dropout_values[1] = zoneout_hidden_received;

        if(ptr_layer_received->type_dropout == LAYER_DROPOUT::NONE
          &&
              (zoneout_cell_received != 0_r
              ||
              zoneout_hidden_received != 0_r))
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::ZONEOUT;

            if(++this->total_dropout_zoneout_layers == 1_UZ)
            {
                if(this->Allocate__Generator__Dropout_Zoneout() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Generator__Dropout_Zoneout()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_zoneout_layers;

                    return false;
                }
                else if(this->Allocate__Block_Unit__Mask_Dropout_Zoneout() == false)
                {
                    ERR(L"An error has been triggered from the \"Allocate__Block_Unit__Mask_Dropout_Zoneout()\" function.",);
                    
                    ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;

                    --this->total_dropout_zoneout_layers;

                    return false;
                }
            }
        }
        else if(zoneout_cell_received == 0_r
                 &&
                 zoneout_hidden_received == 0_r
                 &&
                 ptr_layer_received->type_dropout == LAYER_DROPOUT::ZONEOUT)
        {
            ptr_layer_received->type_dropout = LAYER_DROPOUT::NONE;
            
            if(this->total_dropout_zoneout_layers != 0_UZ && --this->total_dropout_zoneout_layers == 0_UZ)
            {
                this->Deallocate__Generator__Dropout_Zoneout();

                this->Deallocate__Cell_Unit__Mask_Dropout_Zoneout();
            }
        }
    }

    return true;
}

void Model::Scale_Weight__Dropout(real const scale_factor_received, Layer const *const layer_it)
{
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT: this->Scale_Weight__FC__Recurrent__Dropout(scale_factor_received, layer_it);
        case LAYER::FULLY_CONNECTED: this->Scale_Weight__FC__Forward__Dropout(scale_factor_received, layer_it); break;
        default:
            ERR(L"Type layer (%u | %ls) is not managed in the switch.",
                                     layer_it->type_layer,
                                     LAYER_NAME[layer_it->type_layer].c_str());
                return;
    }
}

void Model::Scale_Weight__FC__Forward__Dropout(real const scale_factor_received, Layer const *const layer_it)
{
    Neuron_unit const *const tmp_ptr_layer_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit - 1), // Get last neuron unit.
                                 *const tmp_ptr_layer_ptr_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    var const *const tmp_ptr_array_parameters_end(this->ptr_array_parameters + *tmp_ptr_layer_ptr_last_neuron_unit->ptr_last_connection_index);
    var *tmp_ptr_array_parameters_it(this->ptr_array_parameters + *tmp_ptr_layer_ptr_first_neuron_unit->ptr_first_connection_index);
    
    for(; tmp_ptr_array_parameters_it != tmp_ptr_array_parameters_end; ++tmp_ptr_array_parameters_it) { *tmp_ptr_array_parameters_it *= scale_factor_received; }
}

void Model::Scale_Weight__FC__Recurrent__Dropout(real const scale_factor_received, Layer const *const layer_it)
{
    AF_Ind_recurrent_unit const *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);

    size_t const tmp_number_units(static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_first_AF_Ind_recurrent_unit));
    
    var *tmp_ptr_array_parameters_it(this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    var const *const tmp_ptr_array_parameters_end(tmp_ptr_array_parameters_it + tmp_number_units);
    
    for(; tmp_ptr_array_parameters_it != tmp_ptr_array_parameters_end; ++tmp_ptr_array_parameters_it) { *tmp_ptr_array_parameters_it *= scale_factor_received; }
}
}
