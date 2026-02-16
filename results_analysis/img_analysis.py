import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import zoom

##FFT analysis
def fft_analysis():
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


###intesity histogram

def histo_analysis():
    df_30 = pickle.load(open('../data/test/COLI-194-AER-d200_ms2_30.pkl', 'rb'))

    pred_30 = pickle.load(open('../data/pred/COLI-194-AER-d200_ms2_30_3999.pkl','rb'))[0,0,:,:]


    # train dataset transform
    img_30 = df_30[0][90:512,:]
    zoom_factors = np.array((256,512)) / np.array(img_30.shape)
    img_30 = zoom(img_30, zoom_factors, order=1)  # linear interpolation

    cond_30 = np.log(df_30[1][90:512,:]+1)
    img = cond_30
    hist, bins = np.histogram(img.flatten(), bins=256, range=[np.min(img), np.max(img)])

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.plot(bins[:-1], hist, color='black')
    plt.title('Intensity Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.xlim([np.min(img), np.max(img)])
    plt.grid(True)
    plt.show()


##visual comparison

def visual_analysis():
    df_30 = pickle.load(open('../data/test/COLI-194-AER-d200_ms2_30.pkl', 'rb'))

    pred_30 = pickle.load(open('../data/pred/COLI-194-AER-d200_ms2_30_3999.pkl','rb'))[0,0,:,:]


    # train dataset transform
    img_30 = df_30[0][90:512,:]
    img_30 = img_30/255
    img_30[:,:] = (img_30[:,:] - 1.44) / 1.19
    zoom_factors = np.array((256,512)) / np.array(img_30.shape)
    img_30 = zoom(img_30, zoom_factors, order=1)  # linear interpolation

    cond_30 = np.log(df_30[1][90:512,:]+1)


    plt.imshow(img_30, cmap='magma')  # use 'gray' or any other colormap
    plt.colorbar()                # optional, shows value scale
    plt.title("2D Array as Image COLI 4000")
    plt.axis('off')               # optional, hide axes
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def sweep_compare_magma(img_a, img_b, title="Sweep comparison"):
    assert img_a.shape == img_b.shape, "Images must have same shape"

    h, w = img_a.shape

    # Shared color scale (important!)
    vmin = min(img_a.min(), img_b.min())
    vmax = max(img_a.max(), img_b.max())

    fig, ax = plt.subplots()
    ax.set_title(title)

    split = w // 2
    combined = np.zeros_like(img_a)
    combined[:, :split] = img_a[:, :split]
    combined[:, split:] = img_b[:, split:]

    im = ax.imshow(
        combined,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        origin='upper'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Intensity")

    line = ax.axvline(split, color='cyan', linewidth=1)

    ax.axis('off')

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return

        x = int(np.clip(event.xdata, 0, w))

        combined[:, :x] = img_a[:, :x]
        combined[:, x:] = img_b[:, x:]

        im.set_data(combined)
        line.set_xdata([x, x])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


# # Example usage
# df_30 = pickle.load(open('../data/pred/COLI-54-ANA-d200_ms2_32.pkl', 'rb'))
# img_30 = df_30[0][90:512, :]
# zoom_factors = np.array((256, 512)) / np.array(img_30.shape)
# img_30 = zoom(img_30, zoom_factors, order=1)  # linear interpolation
# pred_30 = pickle.load(open('../data/pred/COLI-54-ANA-d200_ms2_32_99.pkl', 'rb'))[0, :, :]
#
# #epoch 99
# sweep_compare_magma(img_30, pred_30)

df_30 = pickle.load(open('../data/pred/COLI-54-ANA-d200_ms2_32.pkl', 'rb'))
img_30 = df_30[0][90:512, :]
img_30 = img_30-1
cond_30 = np.log(df_30[1][90:512, :]+1)
zoom_factors = np.array((256, 512)) / np.array(img_30.shape)
img_30 = zoom(img_30, zoom_factors, order=1)  # linear interpolation
cond_30 = zoom(cond_30, zoom_factors, order=1)
pred_30 = pickle.load(open('../data/pred/COLI-54-ANA-d200_ms2_32_299.pkl', 'rb'))[0, :, :]

#epoch 299
sweep_compare_magma(cond_30, pred_30)
sweep_compare_magma(img_30, pred_30)

# histo_analysis()