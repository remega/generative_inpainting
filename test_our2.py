import torchvision.transforms as Trans
import os
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
#from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random


def txt_to_matrix(filename, sp='\t'):
  file = open(filename)
  lines = file.readlines()
  rows = len(lines)
  datamat = np.zeros((rows, 4))
  row = 0

  for line in lines:
    line = line.strip().split(sp)
    if len(line)>4:
      line.remove('')
    try:
      datamat[row, :] = line[:]
    except:
      print(line)
    else:
      datamat[row, :] = line[:]
    row += 1

  return datamat,row



class dataset_sal(data.Dataset):
  def __init__(self, opts, dataroot, datasets):
    self.oriimg = []
    self.orisal = []
    self.names = []
    # np.random.seed(opts.randomSeed)
    # torch.manual_seed(opts.randomSeed)
    # torch.cuda.manual_seed_all(opts.randomSeed)
    for dirname in datasets:
      tempdir = os.path.join(dataroot, dirname)
      images = os.listdir(tempdir)
      for x in images:
        curdir = os.path.join(tempdir, x)
        if os.path.isdir(curdir):
          self.oriimg = self.oriimg + [os.path.join(tempdir, x, 'Ori.png')]
          self.orisal = self.orisal + [os.path.join(tempdir, x, 'OriSalmap.png')]
          self.names.append(x)
    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('validation %s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    img = Image.open(self.oriimg[index]).convert('RGB')
    img = self.transforms_img(img)
    tempmap = Image.open(self.orisal[index])
    salmap = self.transforms_salmap(tempmap)
    salmapup = self.transforms_salmapup(tempmap)
    data = {'srcimg':img, 'srcsal':salmap,  'srcsalup':salmapup, 'name': self.names[index]}
    return data

  def __len__(self):
    return self.size

class dataset_saltar(data.Dataset):
  def __init__(self, opts, dataroot, datasets):
    self.oriimg = []
    self.orisal = []
    self.names = []
    # np.random.seed(opts.randomSeed)
    # torch.manual_seed(opts.randomSeed)
    # torch.cuda.manual_seed_all(opts.randomSeed)
    for dirname in datasets:
      tempdir = os.path.join(dataroot, dirname)
      images = os.listdir(tempdir)
      for x in images:
        curdir = os.path.join(tempdir, x)
        if os.path.isdir(curdir):
          gfile = glob.glob(os.path.join(curdir, 'Salmap*.txt'))
          gfile.sort()
          for tartxt in gfile:
            tempname = tartxt[:-4]
            self.oriimg.append(tempname + '_GT.png')
            self.orisal.append(tempname + '.png')
            self.names.append(x)
    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('validation %s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    img = Image.open(self.oriimg[index]).convert('RGB')
    img = self.transforms_img(img)
    tempmap = Image.open(self.orisal[index])
    salmap = self.transforms_salmap(tempmap)
    salmapup = self.transforms_salmapup(tempmap)
    data = {'srcimg':img, 'srcsal':salmap,  'srcsalup':salmapup, 'name': self.names[index]}
    return data

  def __len__(self):
    return self.size

class dataset_salreal(data.Dataset):
  def __init__(self, opts, dataroot):
    self.oriimg = []
    self.orisal = []
    self.names = []
    images = os.listdir(dataroot)
    images.sort()
    for x in images:
      curdir = os.path.join(dataroot, x)
      if os.path.isdir(curdir):
        gfile = glob.glob(os.path.join(curdir, 'Salmap*.jpg'))
        gfile.sort()
        for fulldir in gfile:
          fullname = os.path.split(fulldir)[-1]
          tempname = fullname[7:-4]
          #self.names.append(os.path.split(fulldir)[-2] + '/' + tempname)
          self.names.append(fulldir.split('/')[-3] + '_' + fulldir.split('/')[-2] + '_' + tempname)
          if tempname == 'Ori':
            self.oriimg.append(os.path.join(curdir, 'Ori.jpg'))
            self.orisal.append(os.path.join(curdir, 'Salmap_Ori.jpg'))
          else:
            self.oriimg.append(os.path.join(curdir, 'GT_' + tempname + '.jpg'))
            self.orisal.append(fulldir)
    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('validation %s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    img = Image.open(self.oriimg[index]).convert('RGB')
    img = self.transforms_img(img)
    tempmap = Image.open(self.orisal[index])
    salmap = self.transforms_salmap(tempmap)
    salmapup = self.transforms_salmapup(tempmap)
    data = {'srcimg':img, 'srcsal':salmap,  'srcsalup':salmapup, 'name': self.names[index]}
    return data

  def __len__(self):
    return self.size


class dataset_unpaired(data.Dataset):
  def __init__(self, opts, dataroot, datasets):
    self.oriimg = []
    self.orisal = []
    self.tarsal = []
    self.names = []
    self.boxes = []
    self.opts = opts
    for dirname in datasets:
      tempdir = os.path.join(dataroot, dirname)
      images = os.listdir(tempdir)
      images.sort()
      #print
      for x in images:
          curdir = os.path.join(tempdir, x)
          if os.path.isdir(curdir):
            gfile = glob.glob(os.path.join(curdir, 'Salmap*.txt'))
            gfile.sort()
            for tartxt in gfile:
              tempname = tartxt[:-4] + '.png'
              self.boxes.append(tartxt)
              self.tarsal.append(tempname)
              self.oriimg.append(os.path.join(curdir, 'Ori.png'))
              self.orisal.append(os.path.join(curdir, 'OriSalmap.png'))
              self.names.append(x)

    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('%s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    srcimg = Image.open(self.oriimg[index]).convert('RGB')
    height = srcimg.size[1]
    width = srcimg.size[0]
    srcimg = self.transforms_img(srcimg)
    #tempsrc = Image.open(self.orisal[index])
    tempsrc = cv2.imread(self.orisal[index], cv2.IMREAD_GRAYSCALE)
    tempsrc = cv2.GaussianBlur(tempsrc, (5, 5), 0)
    tempsrc = Image.fromarray(tempsrc.astype("uint8"))
    srcsal = self.transforms_salmap(tempsrc)
    srcsalup = self.transforms_salmapup(tempsrc)
    #temptar = Image.open(self.tarsal[index])
    temptar = cv2.imread(self.tarsal[index], cv2.IMREAD_GRAYSCALE)
    temptar = cv2.GaussianBlur(temptar, (5, 5), 0)
    temptar = Image.fromarray(temptar.astype("uint8"))
    tarsal = self.transforms_salmap(temptar)
    tarsalup = self.transforms_salmapup(temptar)
    objectbox, row = txt_to_matrix(self.boxes[index])
    scales = self.opts.img_resize_size / np.array([height, height, width, width])
    objmask = torch.ones_like(srcsalup)
    oribox = []
    tarbox = []
    if row == 1:
      tempbox = np.rint(objectbox[0,:] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      oribox = tempbox
      tarbox = tempbox
    elif row == 2:
      tempbox = np.rint(objectbox[0,:] * scales)
      tempbox = tempbox.astype(np.int)
      oribox = tempbox
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      tempbox = np.rint(objectbox[1,:] * scales)
      tempbox = tempbox.astype(np.int)
      tarbox = tempbox
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
    else:
      assert(1 == 0)

    data = {'srcimg': srcimg, 'srcsal': srcsal, 'tarsal': tarsal, 'srcsalup': srcsalup, 'tarsalup': tarsalup, 'name': self.names[index], 'objmask': objmask, 'oribox': oribox, 'tarbox': tarbox}
    return data

  def __len__(self):
    return self.size


class dataset_paired(data.Dataset):
  def __init__(self, opts, dataroot, datasets):
    self.oriimg = []
    self.orisal = []
    self.tarsal = []
    self.tarimg = []
    self.boxes = []
    self.names = []
    self.opts = opts
    for dirname in datasets:
      tempdir = os.path.join(dataroot, dirname)
      images = os.listdir(tempdir)
      images.sort()
      for x in images:
          curdir = os.path.join(tempdir, x)
          if os.path.isdir(curdir):
            gfile = glob.glob(os.path.join(curdir, 'Salmap*.txt'))
            gfile.sort()
            for tartxt in gfile:
              tempname = tartxt[:-4]
              self.boxes.append(tartxt)
              self.tarsal.append(tempname + '.png')
              self.oriimg.append(os.path.join(curdir, 'Ori.png'))
              self.orisal.append(os.path.join(curdir, 'OriSalmap.png'))
              self.names.append(x)
              self.tarimg.append(tempname + '_GT.png')

    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('%s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    srcimg = Image.open(self.oriimg[index]).convert('RGB')
    tarimg = Image.open(self.tarimg[index]).convert('RGB')
    height = srcimg.size[1]
    width = srcimg.size[0]
    srcimg = self.transforms_img(srcimg)
    tarimg = self.transforms_img(tarimg)
    #tempsrc = Image.open(self.orisal[index])
    tempsrc = cv2.imread(self.orisal[index], cv2.IMREAD_GRAYSCALE)
    tempsrc = cv2.GaussianBlur(tempsrc, (5, 5), 0)
    tempsrc = Image.fromarray(tempsrc.astype("uint8"))
    srcsal = self.transforms_salmap(tempsrc)
    srcsalup = self.transforms_salmapup(tempsrc)
    #temptar = Image.open(self.tarsal[index])
    temptar = cv2.imread(self.tarsal[index], cv2.IMREAD_GRAYSCALE)
    temptar = cv2.GaussianBlur(temptar, (5, 5), 0)
    temptar = Image.fromarray(temptar.astype("uint8"))
    tarsal = self.transforms_salmap(temptar)
    tarsalup = self.transforms_salmapup(temptar)
    objectbox, row = txt_to_matrix(self.boxes[index])
    scales = self.opts.img_resize_size / np.array([height, height, width, width])
    objmask = torch.ones_like(srcsalup)
    oribox = []
    tarbox = []
    if row == 1:
      tempbox = np.rint(objectbox[0,:] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      oribox = tempbox
      tarbox = tempbox
    elif row == 2:
      tempbox = np.rint(objectbox[0,:] * scales)
      tempbox = tempbox.astype(np.int)
      oribox = tempbox
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      tempbox = np.rint(objectbox[1,:] * scales)
      tempbox = tempbox.astype(np.int)
      tarbox = tempbox
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
    else:
      assert(1 == 0)

    data = {'srcimg': srcimg, 'tarimg':tarimg, 'srcsal': srcsal, 'tarsal': tarsal, 'srcsalup': srcsalup, 'tarsalup': tarsalup, 'name': self.names[index], 'objmask': objmask, 'oribox': oribox, 'tarbox': tarbox}
    return data

  def __len__(self):
    return self.size


class dataset_pairedreal(data.Dataset):
  def __init__(self, opts, dataroot):
    self.oriimg = []
    self.orisal = []
    self.tarsal = []
    self.tarimg = []
    self.oriboxes = []
    self.tarboxes = []
    self.names = []
    self.opts = opts

    images = os.listdir(dataroot)
    images.sort()
    for x in images:
        curdir = os.path.join(dataroot, x)
        if os.path.isdir(curdir):
          gfile = glob.glob(os.path.join(curdir, 'GT*.jpg'))
          gfile.sort()
          for fulldir in gfile:
            fulldir.split('/')
            fullname = fulldir.split('/')[-1]
            tempname = fullname[3:-4]
            self.tarboxes.append(os.path.join(curdir, 'Obj_'+tempname+'.txt'))
            self.oriboxes.append(os.path.join(curdir, 'Obj_'+tempname+'_Ori.txt'))
            self.oriimg.append(os.path.join(curdir, 'Ori.jpg'))
            self.orisal.append(os.path.join(curdir, 'Salmap_Ori.jpg'))
            self.tarimg.append(fulldir)
            self.tarsal.append(os.path.join(curdir, 'Salmap_'+tempname+'.jpg'))
            self.names.append(fulldir.split('/')[-3] + '_' + fulldir.split('/')[-2] + '_' + tempname)

            self.tarboxes.append(os.path.join(curdir, 'Obj_'+tempname+'_Ori.txt'))
            self.tarsal.append(os.path.join(curdir, 'Salmap_Ori.jpg'))
            self.tarimg.append(os.path.join(curdir, 'Ori.jpg'))
            self.oriboxes.append(os.path.join(curdir, 'Obj_' + tempname + '.txt'))
            self.oriimg.append(fulldir)
            self.orisal.append(os.path.join(curdir, 'Salmap_'+tempname+'.jpg'))
            self.names.append(fulldir.split('/')[-3]+'_'+ fulldir.split('/')[-2]+'_'+tempname+'2')
    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('%s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    srcimg = Image.open(self.oriimg[index]).convert('RGB')
    tarimg = Image.open(self.tarimg[index]).convert('RGB')
    height = srcimg.size[1]
    width = srcimg.size[0]
    srcimg = self.transforms_img(srcimg)
    tarimg = self.transforms_img(tarimg)
    #tempsrc = Image.open(self.orisal[index])
    tempsrc = cv2.imread(self.orisal[index], cv2.IMREAD_GRAYSCALE)
    tempsrc = cv2.GaussianBlur(tempsrc, (5, 5), 0)
    tempsrc = Image.fromarray(tempsrc.astype("uint8"))
    srcsal = self.transforms_salmap(tempsrc)
    srcsalup = self.transforms_salmapup(tempsrc)
    #temptar = Image.open(self.tarsal[index])
    temptar = cv2.imread(self.tarsal[index], cv2.IMREAD_GRAYSCALE)
    temptar = cv2.GaussianBlur(temptar, (5, 5), 0)
    temptar = Image.fromarray(temptar.astype("uint8"))
    tarsal = self.transforms_salmap(temptar)
    tarsalup = self.transforms_salmapup(temptar)

    oriobjectbox, orirows = txt_to_matrix(self.oriboxes[index], '  ')
    tarobjectbox, tarrows = txt_to_matrix(self.tarboxes[index], '  ')
    scales = self.opts.img_resize_size / np.array([height, height, width, width])
    objmask = torch.ones_like(srcsalup)
    oribox = np.zeros((self.opts.maxobj,4),np.int)
    tarbox = np.zeros((self.opts.maxobj,4),np.int)
    for row in range(orirows):
      tempbox = np.rint(oriobjectbox[row, :] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      oribox[row, :] = tempbox

    for row in range(tarrows):
      tempbox = np.rint(tarobjectbox[row, :] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      tarbox[row, :] = tempbox


    data = {'srcimg': srcimg, 'tarimg':tarimg, 'srcsal': srcsal, 'tarsal': tarsal, 'srcsalup': srcsalup, 'tarsalup': tarsalup, 'name': self.names[index], 'objmask': objmask, 'oribox': oribox, 'tarbox': tarbox}
    return data

  def __len__(self):
    return self.size


class dataset_pairedreal_ag(data.Dataset):
  def __init__(self, opts, dataroot):
    self.oriimg = []
    self.orisal = []
    self.tarsal = []
    self.tarimg = []
    self.oriboxes = []
    self.tarboxes = []
    self.names = []
    self.opts = opts

    images = os.listdir(dataroot)
    images.sort()
    for x in images:
        curdir = os.path.join(dataroot, x)
        if os.path.isdir(curdir):
          gfile = glob.glob(os.path.join(curdir, 'GT*.jpg'))
          gfile.sort()
          for fulldir in gfile:
            fulldir.split('/')
            fullname = fulldir.split('/')[-1]
            tempname = fullname[3:-4]
            self.tarboxes.append(os.path.join(curdir, 'Obj_'+tempname+'.txt'))
            self.oriboxes.append(os.path.join(curdir, 'Obj_'+tempname+'_Ori.txt'))
            self.oriimg.append(os.path.join(curdir, 'Ori.jpg'))
            self.orisal.append(os.path.join(curdir, 'Salmap_Ori.jpg'))
            self.tarimg.append(fulldir)
            self.tarsal.append(os.path.join(curdir, 'Salmap_'+tempname+'.jpg'))
            self.names.append(fulldir.split('/')[-3] + '_' + fulldir.split('/')[-2] + '_' + tempname)

            self.tarboxes.append(os.path.join(curdir, 'Obj_'+tempname+'_Ori.txt'))
            self.tarsal.append(os.path.join(curdir, 'Salmap_Ori.jpg'))
            self.tarimg.append(os.path.join(curdir, 'Ori.jpg'))
            self.oriboxes.append(os.path.join(curdir, 'Obj_' + tempname + '.txt'))
            self.oriimg.append(fulldir)
            self.orisal.append(os.path.join(curdir, 'Salmap_'+tempname+'.jpg'))
            self.names.append(fulldir.split('/')[-3]+'_'+ fulldir.split('/')[-2]+'_'+tempname+'2')
    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('%s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    srcimg = Image.open(self.oriimg[index]).convert('RGB')
    tarimg = Image.open(self.tarimg[index]).convert('RGB')
    height = srcimg.size[1]
    width = srcimg.size[0]
    srcimg = self.transforms_img(srcimg)
    tarimg = self.transforms_img(tarimg)
    #tempsrc = Image.open(self.orisal[index])
    tempsrc = cv2.imread(self.orisal[index], cv2.IMREAD_GRAYSCALE)
    tempsrc = cv2.GaussianBlur(tempsrc, (5, 5), 0)
    tempsrc = Image.fromarray(tempsrc.astype("uint8"))
    srcsal = self.transforms_salmap(tempsrc)
    srcsalup = self.transforms_salmapup(tempsrc)
    #temptar = Image.open(self.tarsal[index])
    temptar = cv2.imread(self.tarsal[index], cv2.IMREAD_GRAYSCALE)
    temptar = cv2.GaussianBlur(temptar, (5, 5), 0)
    temptar = Image.fromarray(temptar.astype("uint8"))
    tarsal = self.transforms_salmap(temptar)
    tarsalup = self.transforms_salmapup(temptar)

    oriobjectbox, orirows = txt_to_matrix(self.oriboxes[index], '  ')
    tarobjectbox, tarrows = txt_to_matrix(self.tarboxes[index], '  ')
    scales = self.opts.img_resize_size / np.array([height, height, width, width])
    objmask = torch.ones_like(srcsalup)
    oribox = np.zeros((self.opts.maxobj,4),np.int)
    tarbox = np.zeros((self.opts.maxobj,4),np.int)
    for row in range(orirows):
      tempbox = np.rint(oriobjectbox[row, :] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      oribox[row, :] = tempbox

    for row in range(tarrows):
      tempbox = np.rint(tarobjectbox[row, :] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      tarbox[row, :] = tempbox


    data = {'srcimg': srcimg, 'tarimg':tarimg, 'srcsal': srcsal, 'tarsal': tarsal, 'srcsalup': srcsalup, 'tarsalup': tarsalup, 'name': self.names[index], 'objmask': objmask, 'oribox': oribox, 'tarbox': tarbox}
    return data

  def __len__(self):
    return self.size


class dataset_pairedreal_s(data.Dataset):
  def __init__(self, opts, dataroot):
    self.oriimg = []
    self.orisal = []
    self.tarsal = []
    self.tarimg = []
    self.oriboxes = []
    self.tarboxes = []
    self.names = []
    self.opts = opts

    images = os.listdir(dataroot)
    images.sort()
    for x in images:
        curdir = os.path.join(dataroot, x)
        if os.path.isdir(curdir):
          gfile = glob.glob(os.path.join(curdir, 'GT*.jpg'))
          gfile.sort()
          for fulldir in gfile:
            fulldir.split('/')
            fullname = fulldir.split('/')[-1]
            tempname = fullname[3:-4]
            self.tarboxes.append(os.path.join(curdir, 'Obj_'+tempname+'.txt'))
            self.oriboxes.append(os.path.join(curdir, 'Obj_'+tempname+'_Ori.txt'))
            self.oriimg.append(os.path.join(curdir, 'Ori.jpg'))
            self.orisal.append(os.path.join(curdir, 'Salmap_Ori.jpg'))
            self.tarimg.append(fulldir)
            self.tarsal.append(os.path.join(curdir, 'Salmap_'+tempname+'.jpg'))
            self.names.append(fulldir.split('/')[-3] + '_' + fulldir.split('/')[-2] + '_' + tempname)

    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('%s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    srcimg = Image.open(self.oriimg[index]).convert('RGB')
    tarimg = Image.open(self.tarimg[index]).convert('RGB')
    height = srcimg.size[1]
    width = srcimg.size[0]
    srcimg = self.transforms_img(srcimg)
    tarimg = self.transforms_img(tarimg)
    #tempsrc = Image.open(self.orisal[index])
    tempsrc = cv2.imread(self.orisal[index], cv2.IMREAD_GRAYSCALE)
    tempsrc = cv2.GaussianBlur(tempsrc, (5, 5), 0)
    tempsrc = Image.fromarray(tempsrc.astype("uint8"))
    srcsal = self.transforms_salmap(tempsrc)
    srcsalup = self.transforms_salmapup(tempsrc)
    #temptar = Image.open(self.tarsal[index])
    temptar = cv2.imread(self.tarsal[index], cv2.IMREAD_GRAYSCALE)
    temptar = cv2.GaussianBlur(temptar, (5, 5), 0)
    temptar = Image.fromarray(temptar.astype("uint8"))
    tarsal = self.transforms_salmap(temptar)
    tarsalup = self.transforms_salmapup(temptar)

    oriobjectbox, orirows = txt_to_matrix(self.oriboxes[index], '  ')
    tarobjectbox, tarrows = txt_to_matrix(self.tarboxes[index], '  ')
    scales = self.opts.img_resize_size / np.array([height, height, width, width])
    objmask = torch.ones_like(srcsalup)
    oribox = np.zeros((self.opts.maxobj,4),np.int)
    tarbox = np.zeros((self.opts.maxobj,4),np.int)
    for row in range(orirows):
      tempbox = np.rint(oriobjectbox[row, :] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      oribox[row, :] = tempbox

    for row in range(tarrows):
      tempbox = np.rint(tarobjectbox[row, :] * scales)
      tempbox = tempbox.astype(np.int)
      objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
      tarbox[row, :] = tempbox


    data = {'srcimg': srcimg, 'tarimg':tarimg, 'srcsal': srcsal, 'tarsal': tarsal, 'srcsalup': srcsalup, 'tarsalup': tarsalup, 'name': self.names[index], 'objmask': objmask, 'oribox': oribox, 'tarbox': tarbox}
    return data

  def __len__(self):
    return self.size


class dataset_saltest(data.Dataset):
    def __init__(self, opts, dataroot, datasets):
      self.inputdir = []
      self.outputdir = []
      self.names = []
      # np.random.seed(opts.randomSeed)
      # torch.manual_seed(opts.randomSeed)
      # torch.cuda.manual_seed_all(opts.randomSeed)
      for dirname in datasets:
        tempdir = os.path.join(dataroot, dirname)
        outdir = os.path.join(opts.output_dir, dirname)
        if not os.path.exists(outdir):
          os.makedirs(outdir)
        for imgpath in glob.glob(os.path.join(tempdir, '*.jpg')):
          self.inputdir.append(imgpath)
          imgname = os.path.split(imgpath)[-1]
          shortname = imgname[:-4]
          self.outputdir.append(os.path.join(outdir,imgname))
          self.names.append(shortname)
      self.size = len(self.inputdir)
      # setup image transformation
      transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
      transform_list += [Trans.ToTensor()]
      transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
      self.transforms_img = Trans.Compose(transform_list)
      print('validation %s: %d images' % (dataroot, self.size))

    def __getitem__(self, index):
      img = Image.open(self.inputdir[index]).convert('RGB')
      outdir = self.outputdir[index]
      img = self.transforms_img(img)
      name = self.names[index]
      data = {'input': img, 'outdir': outdir, 'name':name}
      return data

    def __len__(self):
      return self.size


class dataset_saltest_temp(data.Dataset):
  def __init__(self, opts, dataroot, datasets):
    self.inputdir = []
    self.outputdir = []
    self.names = []
    # np.random.seed(opts.randomSeed)
    # torch.manual_seed(opts.randomSeed)
    # torch.cuda.manual_seed_all(opts.randomSeed)
    for dirname in datasets:
      tempdir = os.path.join(dataroot, dirname)
      outdir = os.path.join(opts.output_dir, dirname)
      images = os.listdir(tempdir)
      images.sort()
      if not os.path.exists(outdir):
        os.makedirs(outdir)
      for x in images:
          curdir = os.path.join(tempdir, x)
          outdir2 = os.path.join(outdir, x)
          if not os.path.exists(outdir2):
            os.makedirs(outdir2)
          for imgpath in glob.glob(os.path.join(curdir, '*.png')):
            self.inputdir.append(imgpath)
            imgname = os.path.split(imgpath)[-1]
            shortname = imgname[:-4]
            self.outputdir.append(os.path.join(outdir2,imgname))
            self.names.append(shortname)
    self.size = len(self.inputdir)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    print('validation %s: %d images' % (dataroot, self.size))

  def __getitem__(self, index):
    img = Image.open(self.inputdir[index]).convert('RGB')
    outdir = self.outputdir[index]
    img = self.transforms_img(img)
    name = self.names[index]
    data = {'input': img, 'outdir': outdir, 'name': name}
    return data

  def __len__(self):
    return self.size

class dataset_valRebuttal(data.Dataset):
  def __init__(self, opts, dataroot, datasets):
    self.oriimg = []
    self.orisal = []
    self.tarsal = []
    self.tarimg = []
    self.boxes = []
    self.names = []
    self.indexs = []
    self.opts = opts
    for dirname in datasets:
      tempdir = os.path.join(dataroot, dirname)
      images = os.listdir(tempdir)
      images.sort()
      for x in images:
          curdir = os.path.join(tempdir, x)
          if os.path.isdir(curdir):
            gfile = glob.glob(os.path.join(curdir, 'TarSalmap*.png'))
            gfile.sort()
            count = 1
            for tarsalname in gfile:
              self.tarsal.append(tarsalname)
              self.oriimg.append(os.path.join(curdir, 'Ori.png'))
              self.orisal.append(os.path.join(curdir, 'OriSalmap.png'))
              self.names.append(x)
              self.indexs.append(count)
              count = count + 1
    self.size = len(self.oriimg)
    # setup image transformation
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms_img = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmap = Trans.Compose(transform_list)
    transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    transform_list += [Trans.ToTensor()]
    self.transforms_salmapup = Trans.Compose(transform_list)
    print('%s: %d images'%(dataroot, self.size))

  def __getitem__(self, index):
    srcimg = Image.open(self.oriimg[index]).convert('RGB')
    #tarimg = Image.open(self.tarimg[index]).convert('RGB')
    height = srcimg.size[1]
    width = srcimg.size[0]
    srcimg = self.transforms_img(srcimg)
    #tarimg = self.transforms_img(tarimg)
    #tempsrc = Image.open(self.orisal[index])
    tempsrc = cv2.imread(self.orisal[index], cv2.IMREAD_GRAYSCALE)
    tempsrc = cv2.GaussianBlur(tempsrc, (5, 5), 0)
    tempsrc = Image.fromarray(tempsrc.astype("uint8"))
    srcsal = self.transforms_salmap(tempsrc)
    srcsalup = self.transforms_salmapup(tempsrc)
    #temptar = Image.open(self.tarsal[index])
    temptar = cv2.imread(self.tarsal[index], cv2.IMREAD_GRAYSCALE)
    temptar = cv2.GaussianBlur(temptar, (5, 5), 0)
    temptar = Image.fromarray(temptar.astype("uint8"))
    tarsal = self.transforms_salmap(temptar)
    tarsalup = self.transforms_salmapup(temptar)
    # objectbox, row = txt_to_matrix(self.boxes[index])
    # scales = self.opts.img_resize_size / np.array([height, height, width, width])
    # objmask = torch.ones_like(srcsalup)
    # oribox = []
    # tarbox = []
    # if row == 1:
    #   tempbox = np.rint(objectbox[0,:] * scales)
    #   tempbox = tempbox.astype(np.int)
    #   objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
    #   oribox = tempbox
    #   tarbox = tempbox
    # elif row == 2:
    #   tempbox = np.rint(objectbox[0,:] * scales)
    #   tempbox = tempbox.astype(np.int)
    #   oribox = tempbox
    #   objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
    #   tempbox = np.rint(objectbox[1,:] * scales)
    #   tempbox = tempbox.astype(np.int)
    #   tarbox = tempbox
    #   objmask[:, tempbox[0]:tempbox[1], tempbox[2]:tempbox[3]] = 0
    # else:
    #   assert(1 == 0)

    data = {'srcimg': srcimg, 'srcsal': srcsal, 'tarsal': tarsal, 'srcsalup': srcsalup, 'tarsalup': tarsalup, 'name': self.names[index], 'index': self.indexs[index]}
    return data

  def __len__(self):
    return self.size