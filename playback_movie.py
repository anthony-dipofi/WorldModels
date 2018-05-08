import retro
import os
import numpy as np
import pickle
import scipy.misc

files = os.listdir("train_data")
names = "test3"
for i in range(len(files)):
    #movie = retro.Movie('/media/anthony/linDisk/code/OpenAIRetro/openai_retro_comp_wassname-master/outputs/trajectories/bk2/20180423_13-47-41_human/SonicAndKnuckles3-Genesis-SandopolisZone.Act2-0005.bk2')
    movie = retro.Movie("train_data/"+files[i])
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    env.reset()
    obss = []
    rews = []
    acts = []
    while movie.step():
        keys = []
        for j in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(j))
        action = np.array(keys).astype(float)
        obs, rew, _done, _info = env.step(keys)
        #print(obs.shape)
        print(rew,action)#keys)
        #(obs, rew, action)
        obs = scipy.misc.imresize(obs,(64,64))
        obss.append(obs)
        rews.append(rew)
        acts.append(action)
        #print(_obs)
        #env.render()

    obss_o = np.stack(obss, axis=0 )
    rews_o = np.array(rews)
    acts_o = np.stack(acts, axis=0 )
    dat = (obss_o,rews_o,acts_o)
    pickle.dump( dat, open( "data/data_"+names+"_"+files[i][:-4]+".pth", "wb+" ) )
    env.close()
