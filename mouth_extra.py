'''
extract_mouth_batch.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.

Usage:
    python extract_mouth_batch.py [source directory] [pattern] [target directory] [face predictor path]

    pattern: *.avi, *.mpg, etc

Example:
    python scripts/extract_mouth_batch.py evaluation/samples/GRID/ *.mpg TARGET/ common/predictors/shape_predictor_68_face_landmarks.dat

    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
'''


import os, fnmatch, sys, errno
from skimage import io
import cv2
from mouth_ectraction.lipreading_video import Video


SOURCE_PATH = 'input/s1/1.MP4'
SOURCE_EXTS =  'out.avi'
TARGET_PATH = 'output/'

FACE_PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'

def mkdir_p(path):
    try:
        os.makedirs(path)   # 试着创建这个文件夹
    except OSError as exc:  # Python >2.5   # 当文件存在时无法进行创建
        if exc.errno == errno.EEXIST and os.path.isdir(path):  # 判断跟目录是否有这个文件夹，如果有则跳过，无则创建。
            pass
        else:
            raise

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

# def main():
#     for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
#         print ("Processing: {}".format(filepath))
#         video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)
#         filepath_wo_ext = os.path.splitext(filepath)[0]
#         target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
#         mkdir_p(target_dir)
#
#         i = 0
#         for frame in video.mouth:
#             io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)
#             i += 1
# if __name__ == '__main__':
#     main()

for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    print ("Processing: {}".format(filepath))
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)

    filepath_wo_ext = os.path.splitext(filepath)[0]
    print('os.path.splitext(filepath)==',os.path.splitext(filepath))
    ''' splitext进行分离文件路径和对应的扩展名，分离后得到一个二维的list。第一个为文件名，门后面为对应的扩展名（格式）   '''
    target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
    mkdir_p(target_dir)
    i = 0
    for frame in video.mouth:
    	io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)
        # 从0开始进行逐帧编号，并放入相应的文件夹下
    	i += 1