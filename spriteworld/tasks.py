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
"""Tasks for Spriteworld.

Each class in this file defines a task. Namely, contains a reward function and a
success function for Spriteworld.

The reward function maps an iterable of sprites to a float. The success function
maps an iterable of sprites to a bool.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import random
import numpy as np
import six
from sklearn import metrics
from spriteworld import utils

def satisfy(position, bounds):
  if position[0] >= bounds[0] and position[0] <= bounds[2]:
    return position[1] >= bounds[1] and position[1] <= bounds[3]
  return False
      
shapes = {
  'circle': np.array([1,0,0]), 
  'square': np.array([0,1,0]), 
  'triangle': np.array([0,0,1])
  }

colors = {
  'red': np.array([1,0,0,0]), 
  'blue': np.array([0,1,0,0]), 
  'green': np.array([0,0,1,0]),
  'yellow': np.array([0,0,0,1])
}

@six.add_metaclass(abc.ABCMeta)
class AbstractTask(object):
  """Abstract class from which all tasks should inherit."""

  @abc.abstractmethod
  def reward(self, sprites):
    """Compute reward for the given configuration of sprites.

    This reward is evaluated per-step by the Spriteworld environment. See
    Environment.step() in environment.py for usage. Hence if this is a smooth
    function the agent will have shaped reward. Sparse rewards awarded only at
    the end of an episode can be implemented by returning non-zero reward only
    for a desired goal configuration of sprites (see sub-classes below for
    examples).

    Args:
      sprites: Iterable of sprite instances.

    Returns:
      Float reward for the given configuration of sprites.
    """

  @abc.abstractmethod
  def success(self, sprites):
    """Compute whether the task has been successfully solved.

    Args:
      sprites: Iterable of sprite instances.

    Returns:
      Boolean. Whether or not the given configuration of sprites successfully
        solves the task.
    """


class NoReward(AbstractTask):
  """Used for environments that have no task. Reward is always 0."""

  def __init__(self):
    pass

  def reward(self, unused_sprites):
    """Calculate reward from sprites."""
    return 0.0

  def success(self, unused_sprites):
    return False


class FindGoalPosition(AbstractTask):
  """Used for tasks that require moving some sprites to a goal position."""

  def __init__(self,
               filter_distrib=None,
               goal_position=(0.5, 0.5),
               terminate_distance=0.05,
               terminate_bonus=0.0,
               weights_dimensions=(1, 1),
               sparse_reward=False,
               raw_reward_multiplier=50):
    """Construct goal-finding task.

    This task rewards the agent for bringing all sprites with factors contained
    in a filter distribution to a goal position. Rewards are offset to be
    negative, except for a termination bonus when the goal is reached.

    Args:
      filter_distrib: None or instance of
        factor_distributions.AbstractDistribution. If None, all sprites must be
        brought to the goal position. If not None, only sprites with factors
        contained in this distribution must be brought to the goal position.
      goal_position: Position of the goal.
      terminate_distance: Distance from goal position at which to clip reward.
        If all sprites are within this distance, terminate episode.
      terminate_bonus: Extra bonus for getting all sprites within
        terminate_distance.
      weights_dimensions: Weights modifying the contributions of the (x,
        y)-dimensions to the distance to goal computation.
      sparse_reward: Boolean (default False), whether to provide dense rewards
        or only reward at the end of an episode.
      raw_reward_multiplier: Multiplier for the reward to be applied before
        terminate_bonus. Empirically, 50 seems to be a good value.
    """
    self._filter_distrib = filter_distrib
    self._goal_position = np.asarray(goal_position)
    self._terminate_bonus = terminate_bonus
    self._terminate_distance = terminate_distance
    self._sparse_reward = sparse_reward
    self._weights_dimensions = np.asarray(weights_dimensions)
    self._raw_reward_multiplier = raw_reward_multiplier

  def _single_sprite_reward(self, sprite):
    goal_distance = np.sum(self._weights_dimensions *
                           (sprite.position - self._goal_position)**2.)**0.5
    raw_reward = self._terminate_distance - goal_distance
    return self._raw_reward_multiplier * raw_reward

  def _filtered_sprites_rewards(self, sprites):
    """Returns list of rewards for the filtered sprites."""
    rewards = [
        self._single_sprite_reward(s) for s in sprites if
        self._filter_distrib is None or self._filter_distrib.contains(s.factors)
    ]
    return rewards

  def reward(self, sprites):
    """Calculate total reward summed over filtered sprites."""
    reward = 0.

    rewards = self._filtered_sprites_rewards(sprites)
    if not rewards:  # No sprites get through the filter, so make reward NaN
      return np.nan
    dense_reward = np.sum(rewards)

    if all(np.array(rewards) >= 0):  # task succeeded
      reward += self._terminate_bonus
      reward += dense_reward
    elif not self._sparse_reward:
      reward += dense_reward

    return reward

  def success(self, sprites):
    return all(np.array(self._filtered_sprites_rewards(sprites)) >= 0)


class FindGoalTarget(AbstractTask):
  """Used for tasks that require moving some sprites to a goal position."""

  def __init__(self,
               goal_sprite=0,
               terminate_distance=0.05,
               terminate_bonus=0.5,
               raw_reward_multiplier=1,
               target=None):
    """Construct goal-finding task.

    This task rewards the agent for bringing all sprites with factors contained
    in a filter distribution to a goal position. Rewards are offset to be
    negative, except for a termination bonus when the goal is reached.

    Args:
      goal_sprite: index of the sprite to be carried to the goal position, if -1 the goal_sprite is the agent itself
      goal_position: Position of the goal.
      terminate_distance: Distance from goal position at which to clip reward.
        If all sprites are within this distance, terminate episode.
      terminate_bonus: Extra bonus for getting all sprites within
        terminate_distance.
      weights_dimensions: Weights modifying the contributions of the (x,
        y)-dimensions to the distance to goal computation.
      sparse_reward: Boolean (default False), whether to provide dense rewards
        or only reward at the end of an episode.
      raw_reward_multiplier: Multiplier for the reward to be applied before
        terminate_bonus. Empirically, 50 seems to be a good value.
    """
    self._goal_sprite = goal_sprite
    self._terminate_bonus = terminate_bonus
    self._terminate_distance = terminate_distance
    self._raw_reward_multiplier = raw_reward_multiplier
    self.target = target

  def _reward(self, sprites):
    bound_distance = np.array(sprites[-1].bounds) - np.array(sprites[self._goal_sprite].bounds)
    bound_distance = np.abs(bound_distance).min() 
    goal_distance = np.linalg.norm(sprites[-1].position - sprites[self._goal_sprite].position)
    raw_reward = min(self._terminate_distance - goal_distance, 0.5*self._terminate_distance - bound_distance)
    if self.target is not None:
      target_dist = np.linalg.norm(sprites[self.target].position - sprites[self._goal_sprite].position)
      bound_distance = np.array(sprites[self.target].bounds) - np.array(sprites[self._goal_sprite].bounds)
      bound_distance = np.abs(bound_distance).min() 
      raw_reward = self._terminate_distance - target_dist
      #raw_reward += 1.1 * min(self._terminate_distance - target_dist, 0.5*self._terminate_distance - bound_distance)
    return self._raw_reward_multiplier * raw_reward

  def reward(self, sprites):
    """Calculate total reward summed over filtered sprites."""
    reward = 0.
    reward = self._reward(sprites)
    if reward >= 0: reward += self._terminate_bonus

    for i, sprite in enumerate(sprites[1:-1]):
      if self.target is not None:
        if i == self.target: continue
        if np.linalg.norm(sprites[self._goal_sprite].position - sprite.position) <= self._terminate_distance: 
          reward -= 1

      if np.linalg.norm(sprites[-1].position - sprite.position) <= self._terminate_distance:
        reward -= 1

    return reward

  def success(self, sprites):
    success = self._reward(sprites) >= 0
    return success

class Clustering(AbstractTask):
  """Task for cluster by color/shape conditions."""

  def __init__(self,
               cluster_distribs,
               termination_threshold=2.5,
               terminate_bonus=0.0,
               sparse_reward=False,
               reward_range=10, 
               keys=None):
    """Reward depends on clustering sprites based on color/shape.

    We indicate what feature matters for the clustering with the list of
    cluster distribs. We can then compute intra-extra pairwise distances and use
    the Davies-Bouldin clustering metric.

    See https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation for
    some discussion about different metrics.

    Args:
      cluster_distribs: list of factor distributions defining the clusters.
      termination_threshold: Threshold that the metric should pass to terminate
        an episode. Default of 2.5 seems to work well for 2 or 3 clusters.
      terminate_bonus: Extra bonus upon task success.
      sparse_reward: Boolean (default True), whether to provide dense shaping
        rewards or just the sparse ones at the end of an episode.
      reward_range: Scalar, specifies range [-reward_range, 0] we remap the
        rewards to whenever possible.
    """
    self._cluster_distribs = cluster_distribs
    self._num_clusters = len(cluster_distribs)
    self._termination_threshold = termination_threshold
    self._terminate_bonus = terminate_bonus
    self._sparse_reward = sparse_reward
    self._reward_range = reward_range

  def _cluster_assignments(self, sprites):
    """Return index of cluster for all sprites."""
    clusters = -np.ones(len(sprites), dtype='int')
    for i, sprite in enumerate(sprites):
      for c_i, distrib in enumerate(self._cluster_distribs):
        if distrib.contains(sprite.factors):
          clusters[i] = c_i
          break

    return clusters

  def _compute_clustering_metric(self, sprites):
    """Compute the different clustering metrics, higher should be better."""
    # Get positions of sprites, and their cluster assignments
    cluster_assignments = self._cluster_assignments(sprites)
    positions = np.array([sprite.position for sprite in sprites])
    # Ignore objects unassigned to any cluster
    positions = positions[cluster_assignments >= 0]
    cluster_assignments = cluster_assignments[cluster_assignments >= 0]
    return 1. / metrics.davies_bouldin_score(positions, cluster_assignments)

  def reward(self, sprites):
    """Calculate reward from sprites.

    Recommendation: Use Davies-Bouldin, with termination_threshold left to auto.

    Args:
      sprites: list of Sprites.

    Returns:
      Reward, high when clustering is good.
    """
    reward = 0.
    metric = self._compute_clustering_metric(sprites)

    # Low DB index is better clustering
    dense_reward = (metric -
                    self._termination_threshold) * self._reward_range / 2.

    if metric >= self._termination_threshold:  # task succeeded
      reward += self._terminate_bonus
      reward += dense_reward
    elif not self._sparse_reward:
      reward += dense_reward

    return reward

  def success(self, sprites):
    metric = self._compute_clustering_metric(sprites)
    return metric >= self._termination_threshold


class SortingInteractive(Clustering):
  """Used for tasks that require moving some sprites to a goal position."""

  def __init__(self,
               cluster_distribs,
               cluster_type,
               termination_threshold=0.05,
               terminate_bonus=0.5,
               sparse_reward=False,
               reward_range=10,
               raw_reward_multiplier=1):
    """Construct goal-finding task.

    This task rewards the agent for bringing all sprites with factors contained
    in a filter distribution to a goal position. Rewards are offset to be
    negative, except for a termination bonus when the goal is reached.

    Args:
      goal_sprite: index of the sprite to be carried to the goal position, if -1 the goal_sprite is the agent itself
      goal_position: Position of the goal.
      terminate_distance: Distance from goal position at which to clip reward.
        If all sprites are within this distance, terminate episode.
      terminate_bonus: Extra bonus for getting all sprites within
        terminate_distance.
      weights_dimensions: Weights modifying the contributions of the (x,
        y)-dimensions to the distance to goal computation.
      sparse_reward: Boolean (default False), whether to provide dense rewards
        or only reward at the end of an episode.
      raw_reward_multiplier: Multiplier for the reward to be applied before
        terminate_bonus. Empirically, 50 seems to be a good value.
    """
    self._cluster_distribs = cluster_distribs
    self._num_clusters = len(cluster_distribs)
    self._goal_positions = {}
    self._termination_threshold = termination_threshold
    self._terminate_bonus = terminate_bonus
    self._sparse_reward = sparse_reward
    self._reward_range = reward_range
    self._raw_reward_multiplier = raw_reward_multiplier
    self._cluster_type = cluster_type

  def assign_position(self, clusters):
    if self._goal_positions != {}: return
    # bottom_left, bottom_right, top_left, top_right
    positions = [(0, 0, 0.5, 0.5), (0.5, 0, 1, 0.5), (0, 0.5, 0.5, 1), (0.5, 0.5, 1, 1)]
    random.shuffle(positions)
    for c in clusters:
      if not c in self._goal_positions.keys():
        self._goal_positions[c] = positions.pop(random.randint(0, len(positions) - 1))  
    del positions

  def _position_reward(self, sprites):

    reward = 0
    clusters = self._cluster_assignments(sprites)
    self.assign_position(clusters)
    for i, sprite in enumerate(sprites):
      pos_i = self._goal_positions[clusters[i]]
      if not satisfy(sprite.position, pos_i):
        reward -= 1
      
    return self._raw_reward_multiplier * reward

  def reward(self, sprites):
    """Calculate total reward summed over filtered sprites."""
    
    #reward = 0.

    # Clustering reward
    #metric = self._compute_clustering_metric(sprites)
    reward = self._position_reward(sprites[:-1])

    # Low DB index is better clustering
    #dense_reward = (metric -
    #                self._termination_threshold) * self._reward_range / 2.

    if reward >= self._termination_threshold:  # task succeeded
      reward += self._terminate_bonus
    #  reward += dense_reward
    #elif not self._sparse_reward:
    #  reward += dense_reward


    return reward

  def success(self, sprites):
    reward = self._position_reward(sprites[:-1])
    return reward >= self._termination_threshold
    #metric = self._compute_clustering_metric(sprites)
    #return metric >= self._termination_threshold and self._position_reward(sprites[:-1]) >=0


  def goal_vec(self, sprites):
    clusters = self._cluster_assignments(sprites)
    self.assign_position(clusters)
    positions_ = {
      (0, 0, 0.5, 0.5): (1, 0, 0, 0), 
      (0.5, 0, 1, 0.5): (0, 1, 0, 0), 
      (0, 0.5, 0.5, 1): (0, 0, 1, 0), 
      (0.5, 0.5, 1, 1): (0, 0, 0, 1)
    }
    vecs = {}
    if self._cluster_type == "shape":
      for i, sprite in enumerate(sprites):
        if not sprite.shape in vecs.keys():
          vecs[sprite.shape] = positions_[self._goal_positions[clusters[i]]]
      vecs = [np.concatenate([shapes[k],v]) for k, v in vecs.items()]
    else:
      for i, sprite in enumerate(sprites):
        if sprite.c0 >= 0.9: color = 'red'
        elif sprite.c0 >= 0.55 and sprite.c0 <= 0.65: color = 'blue'
        elif sprite.c0 >= 0.27 and sprite.c0 <= 0.37: color = 'green'
        else: color = 'yellow'
        if not color in vecs.keys():
          vecs[color] = positions_[self._goal_positions[clusters[i]]]
      vecs = [np.concatenate([colors[k],v]) for k, v in vecs.items()]
  
    return np.concatenate(vecs)

class MetaAggregated(AbstractTask):
  """Combines several tasks together."""
  REWARD_AGGREGATOR = {
      'sum': np.nansum,
      'max': np.nanmax,
      'min': np.nanmin,
      'mean': np.nanmean
  }
  TERMINATION_CRITERION = {'all': np.all, 'any': np.any}

  def __init__(self,
               subtasks,
               reward_aggregator='sum',
               termination_criterion='all',
               terminate_bonus=0.0):
    """MetaTasks which combines rewards between several subtasks.

    Args:
      subtasks: Iterable of Tasks.
      reward_aggregator: (string) how to combine rewards together. One of
        ('sum', 'max', 'min', 'mean').
      termination_criterion: (string) how to decide when to terminate, given
        subtasks' termination signals. One of ('all', 'any')
      terminate_bonus: Extra bonus for solving all subtasks, combined with
        termination_criterion.
    """
    if reward_aggregator not in MetaAggregated.REWARD_AGGREGATOR:
      raise ValueError('Unknown reward_aggregator. {} not in {}'.format(
          reward_aggregator, MetaAggregated.REWARD_AGGREGATOR))
    if termination_criterion not in MetaAggregated.TERMINATION_CRITERION:
      raise ValueError('Unknown termination_criterion. {} not in {}'.format(
          termination_criterion, MetaAggregated.TERMINATION_CRITERION))

    self._subtasks = subtasks
    self._reward_aggregator = MetaAggregated.REWARD_AGGREGATOR[
        reward_aggregator]
    self._termination_criterion = MetaAggregated.TERMINATION_CRITERION[
        termination_criterion]
    self._terminate_bonus = terminate_bonus

  def reward(self, sprites):
    rewards = self._reward_aggregator(
        [task.reward(sprites) for task in self._subtasks])
    rewards += self._terminate_bonus * self.success(sprites)
    return rewards

  def success(self, sprites):
    return self._termination_criterion(
        [task.success(sprites) for task in self._subtasks])