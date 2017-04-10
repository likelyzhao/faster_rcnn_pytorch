import os 
import xml.etree.ElementTree as ET

IMAGE_ROOT = "/disk2/data/ILSVRC2015/DET/"


def get_satisfied_images():
    image_set_file = os.path.join(IMAGE_ROOT, 'ImageSets', 'DET')
    image_index = []
    bad_image_index = []
    for i in xrange(1,201):  # there are 200 image_set_file for training
        i_image_set_file = os.path.join(image_set_file, \
                                        'train_' + str(i) + '.txt')
        print 'process {}-th training txt: {}'.format(i, i_image_set_file)
        assert os.path.exists(i_image_set_file), \
            'Path does not exist: {}'.format(i_image_set_file)
        with open(i_image_set_file) as f:
            for x in f.readlines():  ## only use positive training samples
                #print x,len(x)
                if len(x)>2:
                    image_name, flag = x.split(' ')
                    if flag.strip() == '1' and image_ratio_check(image_name) == True :
                        #print "satisfied:{}-{}".format(i,image_name)
                        image_index.extend([image_name])
                    else:
                        #print "bad:{}".format(image_name)
                        bad_image_index.extend([image_name])
    return image_index, bad_image_index

def extra_axis(xml_obj):
    bbox = xml_obj.find('bndbox')
    x1 = float(bbox.find('xmin').text)
    y1 = float(bbox.find('ymin').text)
    x2 = float(bbox.find('xmax').text)
    y2 = float(bbox.find('ymax').text)
    return x1, y1, x2, y2

def image_ratio_check(index, image_ratio=[0.462,6.868],bbox_ratio=[0.117,15.5]):
    """
    if the image or bounding boxes are too large or too small,
    they need to be removed.
    [(x1,y1,x2,y2,name),(...)]
    """
    filename = os.path.join(IMAGE_ROOT, 'Annotations', 'DET', 'train', index + '.xml')
    tree = ET.parse(filename)

    size = tree.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    if width/height<image_ratio[0] or width/height>image_ratio[1]:
        return False

    objs = tree.findall('object')
    # Load object bounding boxes into a data frame.
    for obj in objs:
        x1, y1, x2, y2 = extra_axis(obj)
        if y2-y1<=0 or (x2-x1)/(y2-y1)<bbox_ratio[0] or (x2-x1)/(y2-y1)>bbox_ratio[1]:
            return False

    return True

def save_list(image_index, savepath):
    print "save to {}".format(savepath)
    with open(savepath,"w") as f:
        for name in image_index:
            f.write(name)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    image_index, bad_image_index = get_satisfied_images()
    save_list(image_index, os.path.join(IMAGE_ROOT, 'ImageSets', 'DET', 'train_satisfied.txt'))
    save_list(bad_image_index,  os.path.join(IMAGE_ROOT, 'ImageSets', 'DET', 'train_bad.txt'))
