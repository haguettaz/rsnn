# from rsnn.core.neuron import (FIRING_THRESHOLD, REFRACTORY_RESET, InSpike,
#                               Neuron, State)


# def test_neuron_initialization():
#     # Create a neuron with default parameters
#     neuron = Neuron()

#     # Check if the neuron is initialized correctly
#     assert neuron.threshold == FIRING_THRESHOLD
#     assert len(neuron.states) == 0
#     assert len(neuron.f_times) == 0


# def test_neuron_merge_states():
#     # Create a neuron with default parameters
#     neuron = Neuron()

#     # Create some states to merge
#     states = [
#         State(0.0, 1.0, 0.5),
#         State(2.0, 0.3, 0.1),
#         State(1.0, 0.5, 0.2),
#     ]

#     # Merge the states into the neuron
#     neuron.merge_states(states)

#     # Check if the states are merged correctly
#     assert len(neuron.states) == 3
#     assert neuron.states[0].start == 2.0

#     # Create some more states to merge
#     states = [
#         State(1.5, 1.0, 0.5),
#         State(0.5, 0.3, 0.1),
#         State(3.0, 0.5, 0.2),
#         State(6.0, 0.5, 0.6),
#     ]

#     # Merge the states into the neuron
#     neuron.merge_states(states)

#     # Check if the states are merged correctly
#     assert len(neuron.states) == 7
#     assert neuron.states[0].start == 6.0


# def test_neuron_fire():
#     # Create a neuron with default parameters
#     states = [
#         State(0.0, 1.0, 0.5),
#         State(2.0, 0.3, 0.1),
#         State(1.0, 0.5, 0.2),
#     ]
#     neuron = Neuron(states=states)

#     # Fire the neuron at time 1.0
#     neuron.fire(1.0)

#     # Check if the firing time is recorded correctly
#     assert len(neuron.f_times) == 1
#     assert neuron.f_times[0] == 1.0

#     # Check if the states are updated correctly
#     assert len(neuron.states) == 2
#     assert neuron.states[-1].start == 1.0
#     assert neuron.states[-1].c0 == REFRACTORY_RESET

# def test_neuron_clean_states():
#     # Create a neuron with some states
#     states = [
#         State(0.0, 1.0, 0.5),
#         State(4.0, 0.5, 0.7),
#         State(2.0, 0.3, 0.1),
#         State(5.0, 0.5, 0.8),
#         State(3.0, 0.5, 0.6),
#         State(1.0, 0.5, 0.2),
#     ]
#     neuron = Neuron(states=states)

#     # Clean the states at time 1.5
#     neuron.clean_states(1.5)

#     # Check if the states are cleaned correctly
#     assert len(neuron.states) == 5
#     assert neuron.states[-1].start == 1.0

# def test_neuron_next_firing_time():
#     # Create a neuron with some states
#     states = [
#         State(0.0, 0.0, 2.0, dc1=2.0),
#         State(1.0, 0.0, 0.0, dc1=-.25),
#         State(0.5, 0.0, 0.0, dc1=1.0),
#     ]
#     neuron = Neuron(states=states)

#     # Get the next firing time
#     next_time = neuron.next_firing_time()

#     # Check if the next firing time is correct
#     assert next_time == 0.9000963859659488

#     # Add firing time to the neuron, resetting the neuron potential
#     if next_time is not None:
#         neuron.fire(next_time)

#     # Check if the next firing time is None after firing
#     next_time = neuron.next_firing_time()
#     assert next_time is None

# def test_neuron_step():
#     # Create a neuron with some states
#     states = [
#         State(0.0, 0.0, 2.0, dc1=2.0),
#         State(1.0, 0.0, 0.0, dc1=-.25),
#         State(0.5, 0.0, 0.0, dc1=1.0),
#     ]
#     neuron = Neuron(states=states)

#     # Step the neuron at time 0.0
#     next_time = neuron.step(0.0)

#     # Check if the next firing time and states are correct
#     assert len(neuron.states) == 3
#     assert next_time == 0.9000963859659488

#     # Step the neuron at time 0.75
#     next_time = neuron.step(0.75)

#     # Check if the next firing time and states are correct
#     assert len(neuron.states) == 2
#     assert next_time == 0.9000963859659488


# def test_neuron_learn():
#     # Create a neuron with some states
#     states = [
#         State(0.0, 0.0, 2.0, dc1=2.0),
#         State(1.0, 0.0, 0.0, dc1=-.25),
#         State(0.5, 0.0, 0.0, dc1=1.0),
#     ]
#     neuron = Neuron(states=states)

#     # Define firing times and input spikes
#     f_times = [2.0, 3.0]
#     inspikes = [InSpike(4, 1.5), InSpike(42, 0.8), InSpike(1, 2.5), InSpike(4, 0.7)]

#     # Learn from the firing times and input spikes
#     learned_weights = neuron.learn(f_times, inspikes)

#     assert False, "Implement the learning logic and assertions here"

#     # Check if the learned weights are correct
#     # assert len(learned_weights) == 2
#     # assert learned_weights[1] == pytest.approx(1.5)