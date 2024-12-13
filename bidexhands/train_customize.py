import bidexhands as bi
import torch
#import carb

# Set log level to INFO or DEBUG for more detailed logs
#carb.settings.set("/log/level", "info")
#carb.settings.set("/log/level/default", "info")

# If you want even more detailed logs, set it to TRACE
#carb.settings.set("/log/level", "trace")

env_name =   'ShadowHandRubiksCube' #'ShadowHandBottleCap' #'ShadowHandDoorOpenInward' # # # #'ShadowHandOver' #'ShadowHandScissors' #  #'ShadowHandOver' #'ShadowHandDoorOpenInward' 
algo = "ppo"
env = bi.make(env_name, algo)

obs = env.reset()
terminated = False

while not terminated:
    act = torch.tensor(env.action_space.sample()).repeat((env.num_envs, 1))
    obs, reward, done, info, _,_,_,_,_,_ = env.step(act)
