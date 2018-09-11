import os
import datetime

from django.conf import settings
from object_extractor import Extractor, FRONTALFACE_ALT2
import cv2
import numpy as np
import scipy.misc
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms

from .models import RawImage, CropImage, Portrait
from .nets import FCN8s


def handle_image_file(image_file):
    raw_image = RawImage.objects.create(image_file=image_file)
    raw_image.save()

    # Check the size of the input file
    if raw_image.height_field < 800 and raw_image.width_field < 600:
        return 'size'

    # Extract cropped portraits and check the number of extracted objects
    obj_list = extract(raw_image.image_file.path)
    if len(obj_list) == 1:
        img = obj_list[0]
        # Save the extracted one object
        crop_path = save_image('crop_images', raw_image.image_file.path, img)

        # crop_image = CropImage()
        # crop_image.image_file.name = save_path
        # crop_image.raw_image = raw_image
        # crop_image.save()
    else:
        return 'object'

    # Check whether he(she) wear a hat
    if classify_hat(crop_path):
        return 'hat'

    # Check the angle of image

    # Segmentation
    fcn = FCN8s(2)
    bin_image = reshape_image(segmentation(fcn.net, img))
    save_image('bin_images', raw_image.image_file.path, bin_image)

    # Remove and fill the background

    return bin_image


def save_image(root, filepath, obj):
    now = datetime.datetime.now()

    name = now.strftime("%Y%m%d%H%M%S_") + filepath.split('/')[-1]
    root = os.path.join(settings.MEDIA_ROOT, root)

    cv2.imwrite(os.path.join(root, name), obj)

    return os.path.join(root, name)


def make_file_name(image_path, count):
    file = image_path.split('/')[-1]
    file_name = file.split('.')[0] + '_extracted_' + str(count) + '.' + file.split('.')[1]

    return file_name


def extract(image_path,
            size=(600, 800),
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(300, 400),
            cascade_file=FRONTALFACE_ALT2):
    """ Extract the objects from image and return number of objects detected
    image_path -- The path of the image.
    size -- Size of face images (default None - no rescale at all)
    image -- The image (numpy matrix) read by readImage function.
    min_size -- Minimum possible object size. Objects smaller than that are ignored (default (50,50)).
    scale_factor -- Specifying how much the image size is reduced at each image scale (default 1.1).
    min_neighbors -- Specifying how many neighbors each candidate rectangle should have to retain it (default 5).
    cascade_file  -- The path of cascade xml file use for detection (default current value)
    output_directory -- Directory where to save output (default None - same as input image)
    output_prefix -- Prefix of output (default None - the name of input image)
    startCout -- Specifying the starting of the number put into output names (default 0)
    """

    image = cv2.imread(image_path)
    objects = detect(image,
                     scale_factor=scale_factor,
                     min_neighbors=min_neighbors,
                     min_size=min_size,
                     cascade_file=cascade_file)

    count = 0
    obj_list = []
    for (x, y, w, h) in objects:
        count += 1
        obj = image[y - int(0.2*h):y + int(1.2*h), x - int(0.2*w):x + int(1.2*w)]
        if size:
            # obj.shape == (800, 600, 3)
            obj = cv2.resize(obj, size)
        obj_list.append(obj)

    return obj_list


def detect(image,
           min_size=(300, 400),
           scale_factor=1.1,
           min_neighbors=5,
           cascade_file=FRONTALFACE_ALT2):
    """ Return list of objects detected.
    image -- The image (numpy matrix) read by readImage function.
    min_size -- Minimum possible object size. Objects smaller than that are ignored (default (50,50)).
    scale_factor -- Specifying how much the image size is reduced at each image scale (default 1.1).
    min_neighbors -- Specifying how many neighbors each candidate rectangle should have to retain it (default 5).
    cascade_file  -- The path of cascade xml file use for detection (default current value)
    """

    classifier = cv2.CascadeClassifier(cascade_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return classifier.detectMultiScale(gray_image,
                                       scaleFactor=scale_factor,
                                       minNeighbors=min_neighbors,
                                       minSize=min_size)


COLOR_SET = [
    [255, 255, 255], [12, 12, 12], [190, 193, 212], [214, 188, 192],
    [187, 119, 132], [142, 6, 59], [74, 111, 227], [133, 149, 225],
    [181, 187, 227], [230, 175, 185], [224, 123, 145], [211, 63, 106],
    [17, 198, 56], [141, 213, 147], [198, 222, 199], [234, 211, 198],
    [240, 185, 141], [239, 151, 8], [15, 207, 192], [156, 222, 214],
    [213, 234, 231], [243, 225, 235], [246, 196, 225], [247, 156, 212]
]


# img.shape == (800, 600, 3)
def build_image(img):
    MEAN_VALUES = np.array([104.00698793, 116.66876762, 122.67891434])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))
    height, width, _ = img.shape
    img = np.reshape(img, (1, height, width, 3)) - MEAN_VALUES
    return img


def reshape_image(result):
    s = set()
    _, h, w = result.shape
    result = result.reshape(h*w)
    image = []
    for v in result:
        image.append(COLOR_SET[v])
        if v not in s:
            s.add(v)
    image = np.array(image)
    image = np.reshape(image, (h, w, 3))
    scipy.misc.imsave('/home/joonsun/Downloads/result.jpg', image)

    return image


def segmentation(net, img):
    image = build_image(img)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver = tf.train.Saver()
        model_path = os.path.join(settings.BASE_DIR, 'portraitization/pre_trained_seg_model')
        model_file = tf.train.latest_checkpoint(model_path)
        if model_file:
            saver.restore(sess, model_file)
        else:
            raise Exception('Testing needs pre-trained model!')

        feed_dict = {
            net['image']: image,
            net['drop_rate']: 1
        }
        result = sess.run(tf.argmax(net['score'], dimension=3),
                          feed_dict=feed_dict)

    return result


# transforms the data
loader = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image


def classify_hat(iamge_path):
    model_path = os.path.join(settings.BASE_DIR, 'portraitization/pre_trained_hat_model/hat_model.pt')
    resnet_hat = torch.load(model_path, map_location='cpu')

    prev_model = resnet_hat.training
    resnet_hat.eval()
    image = image_loader(iamge_path)
    with torch.no_grad():
        outputs = resnet_hat(image)
        _, preds = torch.max(outputs, 1)

        resnet_hat.train(mode=prev_model)

    return preds # 0: hair, 1: hat
