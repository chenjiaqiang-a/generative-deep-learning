import os
import argparse
import numpy as np
from networks import VAE
import base.config as config

SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64

def import_data(N, M):
    filelist = os.listdir(config.DATA_ROLLOUT_DIR)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    len_filelist = len(filelist)

    if len_filelist > N:
        filelist = filelist[:N]
    if len_filelist < N:
        N = len_filelist
    
    data = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)
    idx = 0
    file_count = 0

    for file in filelist:
        try:
            new_data = np.load(config.DATA_ROLLOUT_DIR + file)['obs']
            data[idx:(idx + M), :, :, :] = new_data
            
            idx += M
            file_count += 1

            if file_count % 50 == 0:
                print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
        except Exception as e:
            print(e)
            print('Skipped {}...'.format(file))
    print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

    return data, N

def main(args):
    new_model = args.new_model
    N = int(args.N)
    M = int(args.time_steps)
    epochs = int (args.epochs)

    vae = VAE()

    if not new_model:
        try:
            vae.set_weights()
        except:
            print('Either set --new_model or ensure {}/vae/weights.h5 exists'.format(config.RUN_FOLDER))
            raise
    
    try:
        data, N = import_data(N, M)
    except:
        print('NO DATA FOUND')
        raise
    print('DATA SHAPE = {}'.format(data.shape))

    for epoch in range(epochs):
        print('EPOCH ' + str(epoch))
        vae.save_weights(os.path.join(config.RUN_FOLDER, 'vae/weights-{}.h5'.format(epoch)))
        vae.train(data)
    vae.save_weights(os.path.join(config.RUN_FOLDER, 'vae/weights.h5'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--N', default=10000, help='number of episodes to use to train')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('--time_steps', type=int, default=300, help='how many timesteps at start of episode?')
    parser.add_argument('--epochs', default=10, help='number of epochs to train for')
    args = parser.parse_args()

    main(args)