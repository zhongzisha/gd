import sys,os,glob,shutil
from natsort import natsorted
import time

data_root = sys.argv[1]
dirs = glob.glob(os.path.join(data_root, '*'))
lines = []
for d in dirs:
    if os.path.isdir(d):
        print(d)
        dname = d.split(os.sep)[-1]
        save_filename = os.path.join(data_root, dname + '.mp4')
        if os.path.exists(save_filename):
            continue
        filenames = natsorted(glob.glob(os.path.join(d, '*.mp4')))
        lines += [line + '\n' for line in filenames]
        lines += ['='*80 + '\n']
        ts_filenames = []
        for i, filename in enumerate(filenames):
            ts_filename = os.path.join(d, '%02d.ts' % i)
            if not os.path.exists(ts_filename):
                cmd = r'ffmpeg -i "%s" -vcodec copy -acodec copy -vbsf h264_mp4toannexb "%s"' % (
                    filename, ts_filename
                )
                os.system(cmd)
            ts_filenames.append(os.path.basename(ts_filename))
            print(filename, ' done')

        time.sleep(1)
        print(ts_filenames)
        os.chdir(d)
        cmd = r'ffmpeg -i "concat:%s" -acodec copy -vcodec copy -absf aac_adtstoasc "%s"' % (
            '|'.join(ts_filenames), save_filename
        )
        print(cmd)
        os.system(cmd)

        time.sleep(3)
        os.system('del *.ts')
        os.chdir(data_root)

        # break

if len(lines):
    with open(os.path.join(data_root, 'log.txt'), 'w') as fp:
        fp.writelines(lines)

