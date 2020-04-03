import argparse, time, os
import imageio

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset

from PIL import Image
import PIL
from numpy import savez_compressed
import cv2
import numpy as np
import shutil
import matplotlib
import torch

def downloadModel():
    # json parse
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # json parse된것 초기화
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    #json파일로 model로드
    solver = create_solver(opt)

    #testset SR한번 후 본격 SR 진행
    shutil.copy('./results/LR/Test/!.png','./results/LR/MyImage/!.png')
    shutil.copy('./results/LR/Test/!.png','./results/LR/MyImage/!!.png')
    SR(solver,opt,model_name)
    os.remove('./results/LR/MyImage/!!.png')
    return solver,opt,model_name
    
def SR(solver,opt,model_name):

    # dataset가져오기-많이 걸리면 0.002
    bm_names =[]
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        start=time.time()
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]  elapsed time: %.4f sec' % (test_set.name(), len(test_set),time.time()-start))
        bm_names.append(test_set.name())

    #Testset개수만큼 SR
    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]"%bm)

        sr_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_time = []
        scale=4;

        need_HR = False if test_loader.dataset.__class__.__name__.find('LRHR') < 0 else True

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_HR=need_HR)

            # 시간측정
            t0 = time.time()
            solver.test()#SR 
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)
            sr_list.append(visuals['SR'])

            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                path_list.append(os.path.basename(batch['HR_path'][0]).replace('HR', model_name))
                print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (iter+1, len(test_loader),
                                                                                       os.path.basename(batch['LR_path'][0]),
                                                                                       psnr, ssim,
                                                                                       (t1 - t0)))
            else:
                path_list.append(os.path.basename(batch['LR_path'][0]))
                print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                           os.path.basename(batch['LR_path'][0]),
                                                           (t1 - t0)))

        if need_HR:
            print("---- Average PSNR(dB) /SSIM /Speed(s) for [%s] ----" % bm)
            print("PSNR: %.2f      SSIM: %.4f      Speed: %.4f" % (sum(total_psnr)/len(total_psnr),
                                                                  sum(total_ssim)/len(total_ssim),
                                                                  sum(total_time)/len(total_time)))
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----" % (bm,
                                                                      sum(total_time)/len(total_time)))

        
        # save SR results for further evaluation on MATLAB
        if need_HR:
            save_img_path = os.path.join('./results/SR/'+degrad, model_name, bm, "x%d"%scale)
        else:
            save_img_path = os.path.join('./results/SR/'+bm, model_name, "x%d"%scale)

        
        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        for img, name in zip(sr_list, path_list):
            s=time.time()
            #matplotlib.image.save(os.path.join(save_img_path, name),img)
            cv2.imwrite(os.path.join(save_img_path, name),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))# 0.609sec 평균 이미지 하나당 0.07sec
            print("NAME: %s DOWNLOAD TIME:%s\n" %(name,time.time()-s))
            #save(os.path.join(save_img_path, name),img) 5.3sec
            #Image.fromarray(img).save(os.path.join(save_img_path, name)) 2.4sec
            #imageio.imwrite(os.path.join(save_img_path, name), img) 9.8sec
            
        print("===> Total Saving SR images of [%s]... Save Path: [%s] Time: %s\n" %(bm, save_img_path,time.time()-s))

    print("==================================================")
    print("===> Finished !!")
    
    
def main():
    solver,opt,model_name=downloadModel()
    for i in range(1):
        SR(solver,opt,model_name)

if __name__ == '__main__':
    main()