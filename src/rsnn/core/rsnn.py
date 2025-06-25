import os
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm.autonotebook import tqdm

from ..optim.optim import (compute_bounded_discrete_weights,
                           compute_bounded_weights)
from ..signals.spike_train import SpikeTrain
from ..signals.utils import sphere_intersection, sphere_intersection_complement


class Neuron:
    def __init__(
            self,
            idx: int,
            num_synapses: int, 
            firing_threshold: float,
            synaptic_decay: Union[float, List[float]], # synaptic time constant c_k, e.g., 5 ms
            somatic_decay: float, # somatic time constant c_0, e.g., 10 ms
            delays_lim: Tuple[float, float],
            hard_refractory_period: float,
            soft_refractory_period: float,
            soft_refractory_weight: float, # weight of the refractoriness feedback loop
            weights_lim: Tuple[float, float],
            weights_lvl: Optional[int] = None,
            rng: Optional[np.random.Generator] = None
    ):
        self.idx = idx
        self.rng = rng or np.random.default_rng()

        self.num_synapses = num_synapses
        self.firing_threshold = firing_threshold
        self.hard_refractory_period = hard_refractory_period
        self.soft_refractory_period = soft_refractory_period
        self.soft_refractory_weight = soft_refractory_weight

        self.weights_min, self.weights_max = weights_lim
        self.weights_lvl = weights_lvl
        self.delays_min, self.delays_max = delays_lim

        self.synaptic_decay = synaptic_decay
        self.somatic_decay = somatic_decay

        if self.somatic_decay == self.synaptic_decay:
            self.gamma = np.exp(1) / self.somatic_decay
        else:
            tmax = (np.log(self.synaptic_decay) - np.log(self.somatic_decay))/(1/self.somatic_decay - 1/self.synaptic_decay)
            self.gamma = 1 / (np.exp(- tmax / self.somatic_decay) - np.exp(- tmax / self.synaptic_decay))

        assert self.weights_min <= self.weights_max
        assert self.delays_min <= self.delays_max

        self.firing_times = np.array([])

        self.sources = []
        self.targets = []

    def refractory_resp(self, t):
        # () just one time point with one firing time, (n,) one time point with n firing times, (n, m) n time points with m firing times
        if (t.ndim < 2):
            if np.any((0 < t) * (t <= self.hard_refractory_period)):
                return -np.inf
            return -self.soft_refractory_weight * np.sum(((t - self.hard_refractory_period) > 0) * np.exp(- (t - self.hard_refractory_period) / self.soft_refractory_period))

        res = -self.soft_refractory_weight * np.sum(((t - self.hard_refractory_period) > 0) * np.exp(- (t - self.hard_refractory_period) / self.soft_refractory_period), axis=-1)
        res[np.any((0 < t) * (t <= self.hard_refractory_period), axis=-1)] = -np.inf
        return res

    def refractory_resp_deriv(self, t):
        if (t.ndim < 2):
            if np.any((0 < t) * (t <= self.hard_refractory_period)):
                return 0
            return self.soft_refractory_weight / self.soft_refractory_period * np.sum(((t - self.hard_refractory_period) > 0) * np.exp(- (t - self.hard_refractory_period) / self.soft_refractory_period))

        res = self.soft_refractory_weight / self.soft_refractory_period * np.sum(((t - self.hard_refractory_period) > 0) * np.exp(- (t - self.hard_refractory_period) / self.soft_refractory_period), axis=-1)
        res[np.any((0 < t) * (t <= self.hard_refractory_period), axis=-1)] = 0
        return res
    
    def impulse_resp(self, t):
        if self.somatic_decay == self.synaptic_decay:
            return self.gamma * np.sum((t > 0) * t * np.exp(1 - t / self.somatic_decay), axis=-1)
        else:
            return self.gamma * np.sum((t > 0) * (np.exp(- t / self.somatic_decay) - np.exp(- t / self.synaptic_decay)), axis=-1)

    def impulse_resp_deriv(self, t):
        if self.somatic_decay == self.synaptic_decay:
            return self.gamma * np.sum((t > 0) * (np.exp(-t / self.somatic_decay) -  t / self.somatic_decay * np.exp(-t/self.somatic_decay)), axis=-1)
        else:
            return self.gamma * np.sum((t > 0) * (- np.exp(- t / self.somatic_decay) / self.somatic_decay + np.exp(- t / self.synaptic_decay) / self.synaptic_decay), axis=-1)

    def potential(self, t):
        if t.ndim == 0:
            y = np.array([self.impulse_resp(t - self.sources[k].firing_times - self.delays[k]) for k in range(self.num_synapses)])
            return self.weights @ y + self.refractory_resp((t - self.firing_times))

        y = np.array([self.impulse_resp(t[:,None] - self.sources[k].firing_times[None,:] - self.delays[k]) for k in range(self.num_synapses)])
        return self.weights @ y + self.refractory_resp((t[:,None] - self.firing_times[None,:]))
    
    def step(self, t, dt, noise_std, tol=1e-6):
        z = self.potential(t)
        if z > self.noisy_firing_threshold:
            # find the firing time by bisection
            ft = t - dt/2
            dt /= 2
            z = self.potential(ft)
            while np.abs(z - self.noisy_firing_threshold) > tol and dt > tol:
                if z > self.noisy_firing_threshold:
                    ft -= dt/2
                else:
                    ft += dt/2
                dt /= 2
                z = self.potential(ft)
                # if dt < 1e-9:
                #     raise ValueError("No convergence in bisection method.")
            self.firing_times = np.append(self.firing_times, ft)
            self.noisy_firing_threshold = self.firing_threshold + self.rng.normal(0, noise_std)


    def memorize(
            self, 
            spike_trains: Union[List[SpikeTrain], SpikeTrain],
            min_slope: float,
            max_level: float,
            eps: float,
            dt: float = 0.1,
            ):

        assert (min_slope > 0) and (max_level < self.firing_threshold)
        assert dt < eps

        # set time constraints
        # C, a, b = np.empty((0, self.num_synapses)), np.empty(0), np.empty(0)

        if isinstance(spike_trains, SpikeTrain):
            spike_trains = [spike_trains]

        a, b, C = np.empty(0), np.empty(0), np.empty((0, self.num_synapses))

        for spike_train in spike_trains:
            silent_times = sphere_intersection_complement(spike_train.firing_times[self.idx], eps, spike_train.duration, dt)
            active_times = sphere_intersection(spike_train.firing_times[self.idx], eps, spike_train.duration, dt)
            
            firing_min = self.firing_threshold - self.refractory_resp((spike_train.firing_times[self.idx][:,None] - spike_train.firing_times[self.idx][None,:]) % spike_train.duration)
            silent_min = np.full_like(silent_times, -np.inf)
            active_min = np.full_like(active_times, min_slope)
            a = np.concatenate((a, firing_min.T, silent_min.T, active_min.T), axis=0)

            firing_max = self.firing_threshold - self.refractory_resp((spike_train.firing_times[self.idx][:,None] - spike_train.firing_times[self.idx][None,:]) % spike_train.duration)
            silent_max = max_level - self.refractory_resp((silent_times[:,None] - spike_train.firing_times[self.idx][None,:]) % spike_train.duration)
            active_max = np.full_like(active_times, np.inf)
            b = np.concatenate((b, firing_max.T, silent_max.T, active_max.T), axis=0)

            firing_y = np.array([self.impulse_resp((spike_train.firing_times[self.idx][:,None] - spike_train.firing_times[self.sources[k].idx][None,:] - self.delays[k]) % spike_train.duration) for k in range(self.num_synapses)])
            silent_y = np.array([self.impulse_resp((silent_times[:,None] - spike_train.firing_times[self.sources[k].idx][None,:] - self.delays[k]) % spike_train.duration) for k in range(self.num_synapses)])
            active_y = np.array([self.impulse_resp_deriv((active_times[:,None] - spike_train.firing_times[self.sources[k].idx][None,:] - self.delays[k]) % spike_train.duration) for k in range(self.num_synapses)])
            C = np.concatenate((C, firing_y.T, silent_y.T, active_y.T), axis=0)

        assert C.shape[0] == a.shape[0] == b.shape[0]
        assert C.shape[0] > 0        

        if self.weights_lvl is None:
            self.weights, status = compute_bounded_weights(
                C,
                a,
                b,
                (self.weights_min, self.weights_max),
                rng=self.rng
                )
        else:
            self.weights, status = compute_bounded_discrete_weights(
                C,
                a,
                b,
                (self.weights_min, self.weights_max),
                self.weights_lvl,
                rng=self.rng
            )
            
        return status
        
class Network:
    def __init__(
            self, 
            num_neurons: int, 
            num_synapses: int, 
            firing_threshold: float,
            synaptic_decay: Union[float, List[float]], # inverse synaptic time constant c_k, e.g., 1/5 ms
            somatic_decay: float, # inverse somatic time constant c_0, e.g., 1/10 ms
            delays_lim: Tuple[float, float],
            hard_refractory_period:float,
            soft_refractory_period: float,
            soft_refractory_weight: float, # weight of the refractoriness feedback loop
            weights_lim: Tuple[float, float],
            weights_lvl: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
            ):
        
        self.num_neurons = num_neurons
        self.rng = rng or np.random.default_rng()

        self.num_synapses = num_synapses
        self.firing_threshold = firing_threshold
        self.synaptic_decay = synaptic_decay
        self.somatic_decay = somatic_decay
        self.delays_lim = delays_lim
        self.hard_refractory_period = hard_refractory_period
        self.soft_refractory_period = soft_refractory_period
        self.soft_refractory_weight = soft_refractory_weight
        self.weights_lim = weights_lim
        self.weights_lvl = weights_lvl

        self.neurons = [
            Neuron(
                l, 
                num_synapses,
                firing_threshold,
                synaptic_decay,
                somatic_decay,
                delays_lim,
                hard_refractory_period,
                soft_refractory_period,
                soft_refractory_weight,
                weights_lim,
                weights_lvl,
                rng
                ) for l in range(self.num_neurons)]
        
        self.connect()
            
    def connect(self):
        # set source neurons (with weights and delays)
        for neuron in self.neurons:
            sources = self.rng.integers(0, self.num_neurons, neuron.num_synapses)
            neuron.sources = [self.neurons[source] for source in sources]
            neuron.delays = self.rng.uniform(neuron.delays_min, neuron.delays_max, neuron.num_synapses)
            neuron.weights = np.zeros(neuron.num_synapses)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(os.path.join(dirname, "delays.npz"), **{f"delays_{neuron.idx}": neuron.delays for neuron in self.neurons})
        np.savez_compressed(os.path.join(dirname, "weights.npz"), **{f"weights_{neuron.idx}": neuron.weights for neuron in self.neurons})
        np.savez_compressed(os.path.join(dirname, "sources.npz"), **{f"sources_{neuron.idx}": np.array([src.idx for src in neuron.sources]) for neuron in self.neurons})
        np.savez_compressed(os.path.join(dirname, "firing_times.npz"), **{f"firing_times_{neuron.idx}": neuron.firing_times for neuron in self.neurons})

    def load(self, dirname):   
        delays = np.load(os.path.join(dirname, "delays.npz"))
        weights = np.load(os.path.join(dirname, "weights.npz"))
        sources = np.load(os.path.join(dirname, "sources.npz"))
        firing_times = np.load(os.path.join(dirname, "firing_times.npz"))

        for neuron in self.neurons:
            neuron.sources = [self.neurons[src_idx] for src_idx in sources[f"sources_{neuron.idx}"]]
            neuron.delays = delays[f"delays_{neuron.idx}"]
            neuron.weights = weights[f"weights_{neuron.idx}"]
            neuron.firing_times = firing_times[f"firing_times_{neuron.idx}"]

    def memorize(
            self, 
            spike_trains: Union[List[SpikeTrain], SpikeTrain],
            min_slope: float,
            max_level: float,
            eps: tuple,
            res: float = 0.1,
            ):
        
        status = []
        for neuron in tqdm(self.neurons, desc="Memorization"):
            status.append(neuron.memorize(spike_trains, min_slope, max_level, eps, res))
        
        return status

    def run(
            self, 
            duration: float, 
            dt: float, 
            noise_std: float,
            ):
        # only consider autonomous neurons, other neurons have a predetermined firing pattern
        autonomous_neurons = [neuron for neuron in self.neurons if neuron.num_synapses > 0]

        for neuron in autonomous_neurons:
            neuron.noisy_firing_threshold = neuron.firing_threshold + neuron.rng.normal(0, noise_std)

        for t in tqdm(np.arange(0, duration+dt, dt), desc="Network Simulation"):
            for neuron in autonomous_neurons:
                neuron.step(t, dt, noise_std)