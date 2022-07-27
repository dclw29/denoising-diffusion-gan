# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer
import torch.nn as nn
import functools
import torch
import numpy as np


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):

    super().__init__()
    self.config = config
    self.not_use_tanh = config.not_use_tanh
    self.act = act = nn.SiLU()
    self.z_emb_dim = z_emb_dim = config.z_emb_dim
    
    self.nf = nf = config.num_channels_dae
    ch_mult = config.ch_mult
    self.num_res_blocks = num_res_blocks = config.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.attn_resolutions
    dropout = config.dropout
    resamp_with_conv = config.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.conditional  # noise-conditional
    fir = config.fir
    fir_kernel = config.fir_kernel
    self.skip_rescale = skip_rescale = config.skip_rescale
    self.resblock_type = resblock_type = config.resblock_type.lower()
    self.progressive = progressive = config.progressive.lower()
    self.progressive_input = progressive_input = config.progressive_input.lower()
    self.embedding_type = embedding_type = config.embedding_type.lower()
    self.num_classes = config.num_classes # number of conditional classes

    init_scale = 0.
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    module_label_embedding = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      #assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      module_label_embedding.append(nn.Linear(embed_dim, nf * 4)) # LSPR changes to include labels
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      module_label_embedding[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      nn.init.zeros_(module_label_embedding[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      module_label_embedding.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      module_label_embedding[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      nn.init.zeros_(module_label_embedding[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)
    elif resblock_type == 'biggan_oneadagn':
      ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    channels = config.num_channels
    if progressive_input != 'none':
      input_pyramid_ch = channels

    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        #print("Num ",1)
        #print(len(modules)-1)
        in_ch = out_ch
        module_label_embedding.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
          #print("Num ",2)
          #print(len(modules)-1)
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
          module_label_embedding.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
          #print("Num ",3)
          #print(len(modules)-1)          
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))
          module_label_embedding.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch)) # can't change dimension here without upsampling
          #print("Num ",3)
          #print(len(modules)-1)          

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          #print("Num ",4)
          #print(len(modules)-1)          
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          #print("Num ",5)
          #print(len(modules)-1)          
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    module_label_embedding.append(ResnetBlock(in_ch=in_ch))
    #print("Num ",6)
    #print(len(modules)-1)
    modules.append(AttnBlock(channels=in_ch))
    #print("Num ",7)
    #print(len(modules)-1)
    modules.append(ResnetBlock(in_ch=in_ch))
    module_label_embedding.append(ResnetBlock(in_ch=in_ch))
    #print("Num ",8)
    #print(len(modules)-1)

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch                           
        module_label_embedding.append(ResnetBlock(in_ch=in_ch,
                                   out_ch=out_ch))                           
        #print("Num ",9)
        #print(len(modules)-1)

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
        #print("Num ",10)
        #print(len(modules)-1)

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            #print("Num ",11)
            #print(len(modules)-1)
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            #print("Num ",12)
            #print(len(modules)-1)            
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            #print("Num ",13)
            #print(len(modules)-1)                                        
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            #print("Num ",14)
            #print(len(modules)-1)            
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            #print("Num ",15)
            #print(len(modules)-1)                                        
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            #print("Num ",16)
            #print(len(modules)-1)            
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            #print("Num ",17)
            #print(len(modules)-1)            
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
          #print("Num ",18)
          #print(len(modules)-1)          
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))
          module_label_embedding.append(ResnetBlock(in_ch=in_ch))
          #print("Num ",19)
          #print(len(modules)-1)          

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      #print("Num ",20)
      #print(len(modules)-1)   
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
      #print("Num ",21)
      #print(len(modules)-1)  

    self.all_modules = nn.ModuleList(modules)
    self.label_modules = nn.ModuleList(module_label_embedding)
    
    mapping_layers = [PixelNorm(),
                      dense(config.nz, z_emb_dim),
                      self.act,]
    for _ in range(config.n_mlp):
        mapping_layers.append(dense(z_emb_dim, z_emb_dim))
        mapping_layers.append(self.act)
    self.z_transform = nn.Sequential(*mapping_layers)
    
    # naiive label embedding LSPR
    #self.label_embedding = nn.Embedding(self.num_classes, embedding_dim=64) # try same as temb

  def forward(self, x, time_cond, z, label):

    # timestep/noise_level embedding; only for continuous training
    zemb = self.z_transform(z)
    #label_embed = self.label_embedding(label)
    
    modules = self.all_modules
    label_modules = self.label_modules

    m_idx = 0
    ml_idx = 0 # for the labels LSPR
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
     
      temb = layers.get_timestep_embedding(timesteps, self.nf) 
      # could try positional embedding for label also?
      lemb = layers.get_timestep_embedding(label, self.nf) # not timestep but try anyway

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      lemb = label_modules[m_idx](lemb) # label_embed
      m_idx += 1
      ml_idx += 1
      temb = modules[m_idx](self.act(temb))
      lemb = label_modules[m_idx](self.act(lemb))
      m_idx += 1
      ml_idx += 1
    else:
      temb = None
      lemb = None

    if not self.config.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        #print("#####")
        #print("Marker 1")
        #print(hs[-1].size())
        #print(m_idx)
        #print(modules[m_idx])
        h = modules[m_idx](hs[-1], temb, zemb)
        #print(h.size())
        h = label_modules[ml_idx](h, lemb, zemb) # LSPR naiive recombine with additional condition
        #print(h.size())
        # just recombine here?
        m_idx += 1
        ml_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:        
          #print("#####")
          #print("Marker 2")
          #print(hs[-1].size())
          #print(m_idx)
          #print(modules[m_idx])
          h = modules[m_idx](hs[-1], temb, zemb)
          #print(h.size())
          # downsample
          h = label_modules[ml_idx](h, lemb, zemb) # LSPR
          #print(h.size())
          m_idx += 1
          ml_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    #print("#####")
    #print("Marker 3")
    #print(h.size())
    #print(m_idx)
    #print(modules[m_idx])
    h = modules[m_idx](h, temb, zemb)
    #print(h.size())
    h = label_modules[ml_idx](h, lemb, zemb) # LSPR
    m_idx += 1
    ml_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    #print("#####")
    #print("Marker 4")
    #print(h.size())
    #print(m_idx)
    #print(modules[m_idx])
    h = modules[m_idx](h, temb, zemb)
    h = label_modules[ml_idx](h, lemb, zemb)
    #print(h.size())
    #h = modules[m_idx_resbigan_same](h, temb, label_embed) # LSPR
    m_idx += 1
    ml_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        #print("#####")
        #print("Marker 5")
        #o = torch.cat([h, hs.pop()], dim=1)
        #print(o.size())
        #print("o")
        #print(m_idx)
        #print(modules[m_idx])
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
        #print(h.size())
        h = label_modules[ml_idx](h, lemb, zemb) # LSPR
        m_idx += 1
        ml_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          #print("#####")
          #print("Marker 6")
          #print(h.size())
          #print(m_idx)
          #print(modules[m_idx])
          h = modules[m_idx](h, temb, zemb)
          #print(h.size())
          h = label_modules[ml_idx](h, lemb, zemb) # LSPR
          #print(h.size())
          m_idx += 1
          ml_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)

    if not self.not_use_tanh:

        return torch.tanh(h)
    else:
        return h
