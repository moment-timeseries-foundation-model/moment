import logging
import warnings
from argparse import Namespace
from math import ceil

import torch
from torch import nn
from transformers import T5Config

from momentfm.data.base import TimeseriesOutputs
from momentfm.models.layers.embed import PatchEmbedding, Patching
from momentfm.models.layers.revin import RevIN
from momentfm.utils.masking import Masking
from momentfm.utils.utils.utils import NamespaceWithDefaults, _update_inputs, _validate_inputs
from momentfm.utils.t5_infini import T5InfiniModel, T5InfiniEncoderModel

logger = logging.getLogger(__name__)


class ForecastingHead(nn.Module):
    def __init__(self, 
                 head_nf: int = 768*64,
                 forecast_horizon: int = 96, 
                 c_out: int = 1,
                 head_dropout: int = 0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon * c_out) # NEW: c_out for loss dimension (potential for probabilistic predictions)
    
    def forward(self, x, input_mask : torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x forecast_horizon]
        """
        x = self.flatten(x)   # x: [batch_size, n_series, n_patches, d_model]
        x = self.linear(x)    # x: [batch_size, n_series, n_patches*d_model]
        x = self.dropout(x)   # x: [batch_size, n_series, horizon*c_out]
        return x

class Long_Forecaster(nn.Module): 

    def __init__(self, config):

        super().__init__()

        self.d_model = config.d_model
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.transformer_type = config.transformer_type

        self.revin = config.revin
        if config.revin:
            self.normalizer = RevIN(
                num_features=config.n_series, 
                affine=config.revin_affine
        )

        self.tokenizer = Patching(
            patch_len=config.patch_len, 
            stride=config.stride,
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model, 
            seq_len=config.input_size,
            patch_len=config.patch_len, 
            stride=config.stride, 
            dropout=config.dropout, 
            add_positional_embedding=True,
            value_embedding_bias=False, 
            orth_gain=1.41,
        )
        self.mask_generator = Masking(mask_ratio=0.0) # no masking for forecasting task

        # Transformer backbone
        self.encoder = self._get_huggingface_transformer(config)
        
        # Prediction Head
        num_patches = (
                (max(config.input_size, config.patch_len) - config.patch_len) 
                // config.stride + 1
        )

        head_nf = config.d_model * num_patches
        self.head = ForecastingHead(
                head_nf,
                config.h, 
                config.c_out,
                config.head_dropout,
            )

    def _get_huggingface_transformer(self, configs):
        ModelClass, EncoderModelClass = T5InfiniModel, T5InfiniEncoderModel
        
        logger.info(f" ModelClass: {ModelClass.__name__}, EncoderModelClass: {EncoderModelClass.__name__}.")
            
        model_config = T5Config.from_pretrained(
            configs.transformer_backbone)

        setattr(model_config, 'infini_channel_mixing', configs.infini_channel_mixing)
        setattr(model_config, 'use_rope', configs.use_rope)
        setattr(model_config, 'max_sequence_length', configs.input_size / configs.patch_len)
        setattr(model_config, 'n_series', configs.n_series)
      
        transformer_backbone = ModelClass(model_config)
        logging.info(f"Initializing randomly initialized\
                       transformer from {configs.transformer_backbone}.  ModelClass: {ModelClass.__name__}.")
        
        transformer_backbone = transformer_backbone.get_encoder() #check valid inputs to raise error if not encoder-only
        
        if configs.getattr('enable_gradient_checkpointing', True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")
        
        return transformer_backbone

    def forward(self,
        *,
        x_enc: torch.Tensor, 
        input_mask: torch.Tensor = None, 
        **kwargs
    ) -> TimeseriesOutputs:
        """
        x_enc : [batch_size x n_series x seq_len]
        input_mask : [batch_size x seq_len]
        """

        batch_size, n_channels, seq_len = x_enc.shape
        input_mask = torch.ones(batch_size, seq_len).to(x_enc.device) # [batch_size, seq_len]

        # Normalization 
        if self.revin:
            x_enc = self.normalizer(x=x_enc, mask=input_mask, mode='norm')
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0) 
        
        # Patching and embedding
        x_enc = self.tokenizer(x=x_enc) # [batch_size x n_series x n_patch x patch_len]
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))
    
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.d_model)) # [batch_size*n_series, n_patch, d_model]
        
        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            mask=input_mask, 
            patch_len=self.patch_len,
            stride=self.stride).repeat_interleave(n_channels, dim=0)  # [batch_size*n_series, n_patch]

        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask) 
        enc_out = outputs.last_hidden_state
        
        enc_out = enc_out.reshape(
            (-1, n_channels, n_patches, self.d_model)) 
        # [batch_size, n_series, n_patch, d_model]

        # Decoder
        dec_out = self.head(enc_out) # [batch_size, n_series, horizon*c_out]
        
        # De-Normalization
        if self.revin:
            dec_out = self.normalizer(x=dec_out, mode='denorm') # [batch_size, n_series, horizon*c_out]

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

class MOMENT(nn.Module):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()

        if isinstance(config, (argparse.Namespace, SimpleNamespace)):
              
        elif isinstance(config, dict):
            config['c_out'] = 1  
        
        config = _update_inputs(config)
        config = _validate_inputs(config)
        setattr(config, 'c_out', 1) #self.loss.outputsize_multiplier --> NEW: c_out for loss dimension (potential for probabilistic predictions)
        self.h = config.h
        self.input_size = config.input_size
        
        # Channel-independent: n_series=1, Channel_dependent/multivariate prediction: n_series=n_series
        if not hasattr(config, 'n_series'):
            raise AttributeError("config is missing required (int) attribute 'n_series'")
        if not hasattr(config, 'infini_channel_mixing'):
            raise AttributeError("config is missing required (bool) attribute 'infini_channel_mixing'")
            
        if config.infini_channel_mixing==False:
            setattr(config, 'n_series', 1)

        if config.task_name == 'forecasting':
            self.model = Long_Forecaster(config)
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")
    
    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:

        #x_enc: [batch_size, n_series, seq_len]
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])
            
        return self.model(x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs) #dec_out: [batch_size, n_series, horizon*c_out]
