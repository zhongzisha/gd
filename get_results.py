import sys,os,glob,cv2
import numpy as np

subset = 'val'
images_dir = '/media/ubuntu/Temp/gd_mc_seg_Aug1/mc_seg_v6_landslide10/images/%s' % subset
images_shown_dir = '/media/ubuntu/Temp/gd_mc_seg_Aug1/mc_seg_v6_landslide10/images_shown/%s' % subset
results_dir = '/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/deeplabv3_unet_s5-d16_512x512_20k_gd_landslide_512_v6_40000/%s_results_new/' % subset
save_dir = '/media/ubuntu/Temp/results_landslide_%s' % subset


subset = 'val'
images_dir = '/media/ubuntu/Temp/gd_mc_seg_Aug1/mc_seg_v6_water6/images/%s' % subset
images_shown_dir = '/media/ubuntu/Temp/gd_mc_seg_Aug1/mc_seg_v6_water6/images_shown/%s' % subset
results_dir = '/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/deeplabv3_unet_s5-d16_512x512_40k_gd_water_512_v6_20000/%s_results_new/' % subset
save_dir = '/media/ubuntu/Temp/results_water_%s' % subset

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


files = glob.glob(images_dir+'/*.jpg')
inds = np.random.choice(np.arange(len(files)), size=40, replace=False)
for ind in inds:
    print(files[ind])
    fileprefix= files[ind].split(os.sep)[-1].replace('.jpg','')
    orig_img = cv2.imread(files[ind])
    orig_shown_img = cv2.imread(os.path.join(images_shown_dir, fileprefix+'.jpg'))
    pred_img = cv2.imread(os.path.join(results_dir, fileprefix+'.jpg'))
    H,W=orig_img.shape[:2]
    new_img = np.concatenate([orig_img, 255*np.ones((H, 3, 3), dtype=np.uint8),
                              orig_shown_img, 255*np.ones((H, 3, 3), dtype=np.uint8),
                              pred_img], axis=1)
    cv2.imwrite(os.path.join(save_dir, fileprefix+'.jpg'), new_img)




























