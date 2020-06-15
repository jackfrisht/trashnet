import os, sys
import glob, shutil

data = 'dataset-resized'
train = 'train'
val = 'val'
test = 'test'

names = glob.glob('dataset-resized/*')

for name in names:
    na = os.path.split(name)[-1]
    try:
        os.makedirs(os.path.join(train, na))
    except:
        print('exist')

    try:
        os.makedirs(os.path.join(val, na))
    except:
        print('exist')

    try:
        os.makedirs(os.path.join(test, na))
    except:
        print('exist')

    path = os.path.join(data, na, '*.jpg')
    imgs = glob.glob(path)

    for img in imgs[:int(len(imgs) * 0.7)]:
        image = os.path.basename(img)
        shutil.copyfile(os.path.join(data, na, image), os.path.join(train, na, image))

    for img in imgs[int(len(imgs) * 0.7):int(len(imgs) * 0.8)]:
        image = os.path.basename(img)
        shutil.copyfile(os.path.join(data, na, image), os.path.join(test, na, image))

    for img in imgs[int(len(imgs) * 0.8):]:
        image = os.path.basename(img)
        shutil.copyfile(os.path.join(data, na, image), os.path.join(val, na, image))



