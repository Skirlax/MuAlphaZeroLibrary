# MuAlphaZeroLibrary

## Introduction
This is a library for training and using the MuZero and AlphaZero algorithms. The following features are currently implmented:
- MuZero and AlphaZero algorithms
  - MuZero paper: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
  - AlphaZero paper: [Mastering the game of Go without human knowledge](https://www.semanticscholar.org/paper/Mastering-the-game-of-Go-without-human-knowledge-Silver-Schrittwieser/c27db32efa8137cbf654902f8f728f338e55cd1c)
- Customizable games and networks
- Training and playing
- Saving and loading models
- Checkpoints and logging
- Parallel self-play
- Parallel hyperparameter search
## 📚 Documentation 📚
To see the project documentation, check the [wiki](https://github.com/Skirlax/MuAlphaZeroLibrary/wiki) page.

## ❗ Get started ❗
### Linux Dependencies
To install the library on Linux, you will need dependencies to build mysqlclient. Check [mysqlclient](https://github.com/PyMySQL/mysqlclient) for a command to install dependencies on your system.
### Python dependencies
The library is built using python3.11, which it is the only tested version. It is recommended that you use the 3.11.* version of python, because of significant speed improvements.

To see the entire list of dependencies, check the requirements.txt file.
### Installation
After installing the dependencies, you can install the library using pip:
```bash 
pip install mu_alpha_zero_library
```
## ⚡ Quick example ⚡
Here is a quick example of how to train a MuZero algorithm to play the atari game of DonkeyKong.

To define our custom game we can subsclass the abstract class MuZeroGame. See examples/donkey_kong.py for an example of how to do this.

Then we can define a MuZeroConfig object to define the hyperparameters of the MuZero algorithm:
```python
from mu_alpha_zero.config import MuZeroConfig

config = MuZeroConfig()
# You can change all the hyperparameters here, for example:
config.num_simulations = 800
```
Finally, we can train the MuZero algorithm:
```python
from mu_alpha_zero import MuZero
from mu_alpha_zero import MuZeroNet
from mu_alpha_zero.mem_buffer import MemBuffer

mz = MuZero(DonkeyKongGame()) # Import your own game.
memory = MemBuffer(config.max_buffer_size)
mz.create_new(config,MuZeroNet,memory,headless=True)
mz.train()
```