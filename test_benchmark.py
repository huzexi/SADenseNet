import argparse
import functools
import json
from os import path
from time import time
import numpy as np
from multiprocessing import Pool
import os
import h5py
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K

from components.datasets import get_dataset
from components.generator import Generator
from components.model import create_model
from components.config import Config
from components.utils import get_dir, path2img_name
from components.utils.evaluate import calc_score_ycrcb_lf, postprocess
from components.utils.render import render_bgr


def test_sample(bgr, ycrcb, name, model_pth, gpuid, config, to_render_bgr, to_render_diff, dataset=''):
    # Set GPU ID and limit the usage of GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    t_config = tf.ConfigProto()
    t_config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=t_config))

    lf = ycrcb[:, :, :, :, 0]

    model = create_model((ycrcb.shape[2], ycrcb.shape[3]), config)
    model.load_weights(model_pth)

    x, _ = Generator.get_xy(lf, config.a_in)
    x = np.expand_dims(x, 0)

    t = time()
    y2 = model.predict(x, batch_size=1)
    y2 = postprocess(y2[0])
    t = time() - t

    render_diff = path.join(get_dir(config.dir_tmp_test, dataset+".diff", name), "%d.png") if to_render_diff else False
    res = calc_score_ycrcb_lf(y2, ycrcb, bgr, a_in=config.a_in, a_out=config.a_out,
                              render_diff=render_diff)

    if to_render_bgr:
        pth_dir = get_dir(config.dir_tmp_test, dataset, path2img_name(name))
        render_bgr(y2, ycrcb, config,
                   pth_dir=pth_dir)
    return res+(t,)


if __name__ == "__main__":
    """Test the samples, if the spatial size doesn't match U-net, it will be cropped."""
    # Parse args
    parser = argparse.ArgumentParser(description='AngularSR Keras test.')
    parser.add_argument('--model', help="Model path.", type=str)
    parser.add_argument('--bgr', help="Render output's BGR version.", action='store_true')
    parser.add_argument('--diff', help="Render diff maps.", action='store_true')
    parser.add_argument('--gpuid', help="ID of the gpu to be used", type=str, default='0')
    parser.add_argument('--mp', help="Number of process", type=int, default=1)
    args = parser.parse_args()

    # Load, predict and evaluate
    # datasets = ['30Scenes', 'reflective', 'occlusions', 'EPFL']
    datasets = ['30Scenes']
    for ds in datasets:
        with h5py.File(get_dataset(ds, Config).get_path_test(), 'r') as h5:
            names = json.loads(h5.attrs['names'])
            n_samples = len(names)

            psnr_lst, ssim_lst, elapse_lst = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
            pbar = tqdm(total=n_samples)

            target = functools.partial(test_sample,
                                       model_pth=args.model, gpuid=args.gpuid, config=Config,
                                       to_render_bgr=args.bgr, to_render_diff=args.diff,
                                       dataset=ds)

            for i in range(0, n_samples, args.mp):
                name_lst = names[i:i + args.mp]
                length = len(name_lst)
                bgr_lst = [h5[names[i + j] + '/bgr'][:] for j in range(length)]
                ycrcb_lst = [h5[names[i + j] + '/ycrcb'][:] for j in range(length)]

                if args.mp > 1:
                    with Pool(args.mp) as pool:
                        res_lst = pool.starmap(target, zip(bgr_lst, ycrcb_lst, name_lst))
                else:
                    res_lst = [target(bgr, ycrcb, name) for bgr, ycrcb, name in zip(bgr_lst, ycrcb_lst, name_lst)]

                psnr_lst[i:i + length] = [res[0] for res in res_lst]
                ssim_lst[i:i + length] = [res[1] for res in res_lst]
                elapse_lst[i:i + length] = [res[2] for res in res_lst]

                pbar.update(len(bgr_lst))

        # Summary
        psnr_score = np.mean(psnr_lst)
        ssim_score = np.mean(ssim_lst)
        elapse = np.mean(elapse_lst)
        print("Final: PSNR: %.2f, SSIM: %.4f, time: %.2f" % (float(psnr_score), float(ssim_score), float(elapse)))

        # Individual sample result
        save_pth = path.join(Config.dir_tmp_test, 'test_model_%s.csv' % ds)
        with open(save_pth, 'w') as f:
            f.write(','.join(['sample', 'psnr', 'ssim', 'time']) + '\n')
            for i in range(len(psnr_lst)):
                f.write(','.join(
                    [path2img_name(names[i]), "%.2f" % psnr_lst[i], "%.4f" % ssim_lst[i], "%.2f" % elapse_lst[i]]
                     ) + '\n')
