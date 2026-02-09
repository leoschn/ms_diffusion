import numpy as np
import matplotlib.pyplot as plt
import pickle

img_pred = pickle.load(open('../data/pred/RAOORN-11-ANA_100vW_100SPD_ms2_30_3999.pkl', 'rb'))
img_pred = img_pred[0,0,:,:]
f_pred = np.fft.fft2(img_pred)
fshift_pred = np.fft.fftshift(f_pred)
magnitude_spectrum_pred = 20 * np.log(np.abs(fshift_pred))
img_ori = pickle.load(open('../data/test/RAOORN-11-ANA_100vW_100SPD_ms2_30.pkl', 'rb'))[0]

img_ori = img_ori[:,:]
f_ori = np.fft.fft2(img_ori)
fshift_ori = np.fft.fftshift(f_ori)
magnitude_spectrum_ori = 20 * np.log(np.abs(fshift_ori))



plt.imshow(magnitude_spectrum_pred, cmap='gray')
plt.title('Magnitude Spectrum pred')
plt.show()

# plt.imshow(magnitude_spectrum_ori, cmap='gray')
# plt.title('Magnitude Spectrum ori')
# plt.show()