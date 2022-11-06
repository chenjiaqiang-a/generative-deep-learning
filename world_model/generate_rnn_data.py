import os
import argparse
import numpy as np
from networks import VAE
import base.config as config


def get_filelist(N):
    filelist = os.listdir(config.DATA_ROLLOUT_DIR)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
        filelist = filelist[:N]
    if length_filelist < N:
        N = length_filelist

    return filelist, N


def encode_episode(vae, episode):
    obs = episode['obs']
    action = episode['action']
    reward = episode['reward']
    done = episode['done']

    done = done.astype(int)
    reward = np.where(reward > 0, 1, 0) * np.where(done == 0, 1, 0)

    mu, log_var, _ = vae.encoder.predict(obs)

    initial_mu = mu[0, :]
    initial_log_var = log_var[0, :]

    return (mu, log_var, action, reward, done, initial_mu, initial_log_var)


def main(args):
    N = args.N

    vae = VAE()
    try:
        vae.set_weights(os.path.join(config.RUN_FOLDER, 'vae/weights.h5'))
    except Exception as e:
        print(e)
        print('{} does not exist - ensure you have run 02_train_vae.py first'.format(
            os.path.join(config.RUN_FOLDER, 'vae/weights.h5')))
        raise

    filelist, N = get_filelist(N)

    file_count = 0
    initial_mus = []
    initial_log_vars = []

    for file in filelist:
        try:
            rollout_data = np.load(config.DATA_ROLLOUT_DIR + file)
            mu, log_var, action, reward, done, initial_mu, initial_log_var = encode_episode(
                vae, rollout_data)

            np.savez_compressed(config.DATA_SERIES_DIR + file, mu=mu,
                                log_var=log_var, action=action, reward=reward, done=done)
            initial_mus.append(initial_mu)
            initial_log_vars.append(initial_log_var)

            file_count += 1

            if file_count % 50 == 0:
                print('Encoded {} / {} episodes'.format(file_count, N))
        except Exception as e:
            print(e)
            print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(file_count, N))

    initial_mus = np.array(initial_mus)
    initial_log_vars = np.array(initial_log_vars)
    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_mus.shape))

    np.savez_compressed(config.DATA_DIR + '/initial_z.npz',
                        initial_mu=initial_mus, initial_log_var=initial_log_vars)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate RNN data')
    parser.add_argument('--N', default=10000,
                        help='number of episodes to use to train')
    args = parser.parse_args()

    main(args)
