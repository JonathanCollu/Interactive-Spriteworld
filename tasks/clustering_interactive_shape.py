# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Clustering task used in COBRA.

Cluster sprites by color.

We use 4 types of sprites, based on their hue.
We then compute a Davies-Bouldin clustering metric to assess clustering quality
(and generate a reward). The Clustering task uses a threshold to terminate an
episode when the clustering metric is good enough.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import tasks
from spriteworld import action_spaces
from spriteworld.configs.cobra import common

# Task Parameters
NUM_SPRITES_PER_CLUSTER = 2
MAX_EPISODE_LENGTH = 100

CLUSTERS_DISTS_SHAPE = {
    'circle': distribs.Discrete('shape', ['circle']),
    'square': distribs.Discrete('shape', ['square']),
    'triangle': distribs.Discrete('shape', ['triangle'])
}

# Define train/test generalization splits

MODES_SHAPE = {
    'train': ('triangle', 'square'),
    'test': ('square', 'circle')
}


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  # Select clusters to use, and their c0 factor distribution.
  c0_clusters_shape = [CLUSTERS_DISTS_SHAPE[cluster] for cluster in MODES_SHAPE[mode]]
  print('Clustering task: {}, #sprites: {}'.format(MODES_SHAPE[mode], NUM_SPRITES_PER_CLUSTER))

  other_factors_shape = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('scale', [0.13]),
      distribs.Continuous('c0', 0., 0.9),
      distribs.Continuous('c1', 0.3, 1.),
      distribs.Continuous('c2', 0.9, 1.),
  ])

  # Generate the sprites to be used in this task, by combining Hue with the
  # other factors.

  sprite_factors_shape = [distribs.Product((other_factors_shape, c0)) for c0 in c0_clusters_shape]
  # Convert to sprites, generating the appropriate number per cluster.

  sprite_gen_per_cluster_shape = [
      sprite_generators.generate_sprites(
          factors, num_sprites=NUM_SPRITES_PER_CLUSTER)
      for factors in sprite_factors_shape
  ]
  # Concat clusters into single scene to generate.
  sprite_gen_shape = sprite_generators.chain_generators(*sprite_gen_per_cluster_shape)
  sprite_gen_shape = sprite_generators.shuffle(sprite_gen_shape)

  # Create the agent body
  agent_body_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('shape', list(MODES_SHAPE[mode])),
      distribs.Discrete('scale', [0.07]),
      distribs.Discrete('c0', [1.]),
      distribs.Discrete('c1', [0.]),
      distribs.Discrete('c2', [1.]),
  ])
  agent_body_gen = sprite_generators.generate_sprites(agent_body_factors, num_sprites=1)
  sprite_gen_shape = sprite_generators.chain_generators(sprite_gen_shape, agent_body_gen)
  # Randomize sprite ordering to eliminate any task information from occlusions
  
  task = tasks.Clustering(c0_clusters_shape, terminate_bonus=0., reward_range=10.)

  config = {
      'task': task,
      'action_space': action_spaces.Embodied(step_size=0.05),
      'renderers': common.renderers(),
      'init_sprites': sprite_gen_shape,
      'max_episode_length': MAX_EPISODE_LENGTH,
      'metadata': {
          'name': os.path.basename(__file__),
          'mode': mode
      }
  }
  return config
