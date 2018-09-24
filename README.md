# WorldModels
This is an implementation of the Variational Autoencoder and the Mixture Density Network RNN as described in https://worldmodels.github.io/. This implementation was created for use in the Gotta Learn Fast! reinforcement learning competition (https://arxiv.org/abs/1804.03720). Given trajectories in a game environment, typically in the form of video, the network learns to project future trajectories through the environment. 

The Variation Autoencoder (VAE) creates a compressed representation of each individual frame to capture spatial information, and the RNN is used to capture information about the temporal dynamics. The Mixture Density Network, which uses as its input the outputs of the RNN allows the network to better model the branching of possibilites in its projection of the future.

Use playback_movie.py to play back a saved video of a retro environment run from record_humans_play.py (https://gitlab.com/maglearn/openai_retro_comp_wassname) or some other source, and save the game observations, rewards and actions as a tuple of numpy arrays, which are then loaded by train_VAE.py. 
