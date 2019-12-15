import mnist
import numpy as np
import scipy.ndimage

def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img
    
def apply_noise(img, bit_flip_ratio):
    img = np.array(img)
    mask = np.random.random(size=(28,28)) < bit_flip_ratio
    #base = np.random.random(img[mask].shape)*255
    #img[mask] = base - img[mask]
    img[mask] = 255 - img[mask]
    return img

def get_rotations(img, rotation_steps):
    for rot in rotation_steps:
        img = scipy.ndimage.rotate(img, rot, reshape=False)
        yield img

def binarize(img):
    return np.array(img) > 127

def heal_image(img, seq_len, square_count, square_size, noise_ratio):
    squares_begin = np.random.randint(0, seq_len - square_count)
    squares_end = squares_begin + square_count

    rotations = []
    rotations.append(apply_noise(img,noise_ratio))
    targets = []
    targets.append(img)
    rotation_steps = np.random.random(size=seq_len) * 180 - 90

    for idx, rotation in enumerate(get_rotations(img, rotation_steps)):
        rotations.append(apply_noise(rotation, noise_ratio))
        targets.append(rotation)

 
    return binarize(rotations), binarize(targets)

class HealingMNIST():
    def __init__(self, seq_len=5, square_count=3, square_size=5, noise_ratio=0.15, digits=range(10),test = False):
        mnist_train = [img for img, label in zip(mnist.train_images(), mnist.train_labels()) if label in digits]
        mnist_test = [img for img, label in zip(mnist.test_images(), mnist.test_labels()) if label in digits]

        train_images = []
        train_targets = []
        test_images = []
        test_targets = []

        
        i = 0;
        for img in mnist_train:
            train_img, train_tar = heal_image(img, seq_len, square_count, square_size, noise_ratio)
            train_images.append(train_img)
            train_targets.append(train_tar)
            i = i + 1
            if test:
                if i > 100:
                    break
                
        i = 0;
        for img in mnist_test:
            test_img, test_tar = heal_image(img, seq_len, square_count, square_size, noise_ratio)
            test_images.append(test_img)
            test_targets.append(test_tar)
            i = i + 1
            if test:
                if i > 100:
                    break
        
        self.train_images = np.array(train_images)
        self.train_targets = np.array(train_targets)
        self.test_images = np.array(test_images)
        self.test_targets = np.array(test_targets)
