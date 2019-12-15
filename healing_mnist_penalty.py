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

def get_rotations(img, rotation_steps, lag):
    imgs = []
    for i in range(len(rotation_steps)):
        if i == 0:
            rot = rotation_steps[i];
        else:
            rot = rotation_steps[i] + lag * rotation_steps[i-1];
        img = scipy.ndimage.rotate(img, rot, reshape=False)
        imgs.append(img)
    return imgs

def binarize(img):
    return np.array(img) > 127

def heal_image(img, seq_len, square_count, square_size, noise_ratio, lag):
    squares_begin = np.random.randint(0, seq_len - square_count)
    squares_end = squares_begin + square_count

    rotations = []
    rotations.append(apply_noise(img,noise_ratio))
    targets = []
    rotation_steps = np.random.random(size=seq_len) * 180 - 90

    for rotation in (get_rotations(img, rotation_steps, lag)):
        rotations.append(apply_noise(rotation, noise_ratio))
        targets.append(rotation)

    del rotations[-1]
    return binarize(rotations), binarize(targets), rotation_steps

class HealingMNIST():
    def __init__(self, seq_len=5, square_count=3, square_size=5, noise_ratio=0.15, digits=range(10),test = False, lag=0.1):
        mnist_train = [img for img, label in zip(mnist.train_images(), mnist.train_labels()) if label in digits]
        mnist_test = [img for img, label in zip(mnist.test_images(), mnist.test_labels()) if label in digits]

        train_images = []
        train_targets = []
        test_images = []
        test_targets = []
        train_rotations = []
        test_rotations = []
        
        i = 0;
        for img in mnist_train:
            train_img, train_tar, train_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio, lag=lag)
            train_images.append(train_img)
            train_targets.append(train_tar)
            train_rotations.append(train_rot)
            i = i + 1
            if test:
                if i > 100:
                    break
                
        i = 0;
        for img in mnist_test:
            test_img, test_tar, test_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio, lag=lag)
            test_images.append(test_img)
            test_targets.append(test_tar)
            test_rotations.append(test_rot)
            i = i + 1
            if test:
                if i > 100:
                    break
        
        self.train_images = np.array(train_images)
        self.train_targets = np.array(train_targets)
        self.test_images = np.array(test_images)
        self.test_targets = np.array(test_targets)
        self.train_rotations = np.array(train_rotations)
        self.test_rotations = np.array(test_rotations)