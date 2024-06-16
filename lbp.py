import numpy as np
import math


def get_pixel_bilinear(image, y, x):
    xmin, xmax = math.floor(x), math.ceil(x)
    ymin, ymax = math.floor(y), math.ceil(y)
    intensity_top_left = image[ymin, xmin]
    intensity_top_right = image[ymin, xmax]
    intensity_bottom_left = image[ymax, xmin]
    intensity_bottom_right = image[ymax, xmax]
    
    weight_x = x - xmin
    weight_y = y - ymin
    
    intensity_at_top = (1-weight_x) * intensity_top_left + weight_x * intensity_top_right
    intensity_at_bottom= (1-weight_x) * intensity_bottom_left + weight_x * intensity_bottom_right
    
    final_intensity = (1-weight_y) * intensity_at_top + weight_y * intensity_at_bottom        
    return final_intensity

def lbp(image, P, R, method='bilinear'):
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(R, height - R):
        for j in range(R, width - R):
            center = image[i, j]
            binary_string = ''
            
            for p in range(P):
                theta = (2 * np.pi * p) / P
                x = i + R * np.sin(theta)
                y = j + R * np.cos(theta)
                if method == 'bilinear':
                    value = get_pixel_bilinear(image, y, x)
                else:
                    xx = round(x)
                    yy = round(y)
                    value = image[yy, xx] # nearest pixel
                                         
                binary_string += '1' if value >= center else '0'
            
            lbp_value = int(binary_string, 2)
            lbp_image[i, j] = lbp_value
    
    return lbp_image


# Rotation Invariant LBP, RI-LBP
def ri_lbp(image, P, R):
    lbp = lbp(image, P, R)
    rows, cols = lbp.shape
    ri_lbp = np.zeros_like(lbp)
    
    for i in range(R, rows-R):
        for j in range(R, cols-R):
            binary_string = np.unpackbits(np.array([lbp[i, j]], dtype=np.uint8))
            min_value = float('inf')
            for k in range(P):
                rotated_string = np.roll(binary_string, k)
                decimal_value = np.packbits(rotated_string)[0]
                if decimal_value < min_value:
                    min_value = decimal_value
            ri_lbp[i, j] = min_value
    
    return ri_lbp


# uniform lpb, ULBP
def get_pixel_value(img, center, x, y):
    if img[x, y] >= center:
        return 1
    return 0

def uniform_pattern(value):
    transitions = 0
    bin_str = format(value, '08b')
    for i in range(len(bin_str) - 1):
        if bin_str[i] != bin_str[i + 1]:
            transitions += 1
    return transitions <= 2

def ulbp(image, P, R):
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape
    angle = 2 * np.pi / P

    for i in range(R, height - R):
        for j in range(R, width - R):
            center = image[i, j]
            lbp_value = 0

            for p in range(P):
                theta = p * angle
                x = i + int(R * np.sin(theta))
                y = j + int(R * np.cos(theta))
                lbp_value += get_pixel_value(image, center, x, y) << p

            if uniform_pattern(lbp_value):
                lbp_image[i, j] = lbp_value
            else:
                lbp_image[i, j] = P + 1

    return lbp_image


# Adaptive Local Binary Pattern (ALBP): https://arxiv.org/pdf/2404.14560
def albp(image, P, R, B=0.1, method='bilinear'):
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(R, height - R):
        for j in range(R, width - R):
            center = image[i, j]
            binary_string = ''
            LBV = center * (1 - B)
            RBV = center * (1 + B)
            for p in range(P):
                theta = (2 * np.pi * p) / P
                x = i + R * np.sin(theta)
                y = j + R * np.cos(theta)
                if method == 'bilinear':
                    value = get_pixel_bilinear(image, y, x)
                else:
                    xx = round(x)
                    yy = round(y)
                    value = image[yy, xx] # nearest pixel  
                    
                binary_string += '1' if LBV <= value <= RBV else '0'
            
            lbp_value = int(binary_string, 2)
            lbp_image[i, j] = lbp_value
    
    return lbp_image