from gym import spaces
import numpy as np
import torch
import yaml
import pandas as pd
from rl_games.algos_torch.players import BasicPpoPlayerContinuous, BasicPpoPlayerDiscrete


config_name = "/home/axelcoulon/projects/OmniIsaacGymEnvs/omniisaacgymenvs/cfg/train/JetbotPPO.yaml"

with open(config_name, 'r') as stream:
    cfg = yaml.safe_load(stream)

observation_space = spaces.Box(np.ones(2) * -np.Inf, np.ones(2) * np.Inf)

act_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
print()

player = BasicPpoPlayerContinuous(cfg, observation_space, act_space, clip_actions=True, deterministic=True)
player.restore("/home/axelcoulon/projects/OmniIsaacGymEnvs/runs/Jetbot/save_64_64/nn/Jetbot.pth")

# Chemin vers le fichier Excel
chemin_fichier = "/home/axelcoulon/projects/OmniIsaacGymEnvs/player_observations.xlsx"

# Charger le fichier Excel dans un DataFrame
donnees = pd.read_excel(chemin_fichier)

headings=donnees["Heading"]
distances=donnees["Distance"]

for i in range(len(headings)):
    obs = dict({'obs':torch.tensor([headings[i],distances[i]], dtype=torch.float32, device='cuda')})
    action = player.get_action(obs["obs"], is_deterministic=True)
    print("itteration: ", i)
    print("obs: ",obs)
    print("actions: ",action)
    print("")

