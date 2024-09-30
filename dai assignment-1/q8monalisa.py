import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('Monalisa.jpg')
cc_array = []
if img.ndim == 3:
    img_gray = img[:,:,0] * 0.2989 + img[:,:,1] * 0.5870 + img[:,:,2] * 0.1140
def shift(img_gray,tx):
    shifted_image = np.zeros(img_gray.shape)
    if tx>0:
        shifted_image[:,tx:] = img_gray[:,:-tx]
    elif tx<0:
        shifted_image[:,:tx] = img_gray[:,-tx:]
    else:
        shifted_image = img_gray
    return shifted_image

def correlation(image1,image2):
    img1 = image1.reshape(-1)
    img2 = image2.reshape(-1)
    correlation_coefficient = np.corrcoef(img1,img2)[0,1]
    return correlation_coefficient

for tx in range(-10,11):
    shifted_img = shift(img_gray,tx)
    CC = correlation(img_gray,shifted_img)
    cc_array.append(CC)

plt.plot(range(-10,11),cc_array,marker = 'o')
plt.xlabel("tx")
plt.ylabel("Correlation Coefficients")
plt.savefig('q8_correlation.png')
img_1D = img_gray.reshape(-1)
histogram = np.zeros(256,dtype = int)
for x in img_1D:
    histogram[int(x)]+=1
s=0
for i in histogram:
    s+=i
normalized_histogram = histogram/s
plt.bar(range(256),normalized_histogram,width = 1,edgecolor = 'black')
plt.xlabel('Intensity')
plt.ylabel('Normalized frequency')
plt.ylim(0,0.04)
plt.savefig('q8_histogram.png')
