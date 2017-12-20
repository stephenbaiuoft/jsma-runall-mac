import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from cleverhans.utils_mnist import data_mnist

np_jsma_data_path = 'saver/data_comparison/'
relative_path_2515 = '/home/stephen/PycharmProjects/jsma-runall-mac/cleverhans_tutorials/' \
                     'bai work/CSC2515 Files/'


# JSMA data
# np.savez(relative_path_2515 + np_jsma_data_path + 'jsma_sample.npz', jsma_x=jsma_x)
# # saving training sample
# np.savez(relative_path_2515 + np_jsma_data_path + 'training_sample.npz', sample_x=X_train)

# plots sample data for JSMA and clean data
def plot_jsma_sample():
    data = np.load(relative_path_2515 + np_jsma_data_path + 'jsma_sample.npz')
    jsma_x = data['jsma_x']

    data = np.load(relative_path_2515 + np_jsma_data_path + 'training_sample.npz')
    training_x = data['sample_x']

    # choose which one ==> generated 0-4
    # chose 0, 4
    i = 4
    x = training_x[i].reshape(28,28)
    j_x = jsma_x[i].reshape(28,28)

    plt.figure(1)
    plt.imshow(x, cmap='gray')
    plt.title("Clean Example")

    plt.figure(2)
    plt.imshow(j_x, cmap='gray')
    plt.title("Adversarial Example")
    plt.show()


# show detailed 2 figure comparisons
def test():
    i = 0
    np_jsma_data_path = 'saver/numpy_jsma_x_data/'

    train_end = 3000
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=train_end,
                                                  test_start=0,
                                                  test_end=1)

    training_x = X_train

    data = np.load(relative_path_2515 + np_jsma_data_path + 'jsma_training_x.npz')
    jsma_x = data['training_jsma_x']


    clean_x_u = training_x[i].reshape(28, 28)
    clean_x_5 = pca_filter_img(training_x, 5)[i]
    clean_x_6 = pca_filter_img(training_x, 6)[i]
    clean_x_7 = pca_filter_img(training_x, 7)[i]
    clean_x_10 = pca_filter_img(training_x, 10)[i]

    # 5, 6, 7, 10
    jsma_x_u = jsma_x[i].reshape(28, 28)
    jsma_x_5 = pca_filter_img(jsma_x, 5)[i]
    jsma_x_6 = pca_filter_img(jsma_x, 6)[i]
    jsma_x_7 = pca_filter_img(jsma_x, 7)[i]
    jsma_x_10 = pca_filter_img(jsma_x, 10)[i]


    plt.figure(1)
    plt.imshow(clean_x_u, cmap='gray')
    plt.title('Clean Unfiltered Data')

    plt.figure(2)
    plt.imshow(jsma_x_u, cmap='gray')
    plt.title('Adversarial Unfiltered Data')

    plt.figure(3)

    plt.imshow(clean_x_6, cmap='gray')
    plt.title('Clean Filtered Data')

    plt.figure(4)
    plt.imshow(jsma_x_6, cmap='gray')

    plt.title('Adversarial Filtered Data, n = 36')



    plt.show()

# this plots clean, vs 25, 36, 49, 100 pca filtered samples
def plot_pca_sample():
    # which digit to choose from
    i = 0
    np_jsma_data_path = 'saver/numpy_jsma_x_data/'

    train_end = 3000
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=train_end,
                                                  test_start=0,
                                                  test_end=1)

    training_x = X_train

    data = np.load(relative_path_2515 + np_jsma_data_path + 'jsma_training_x.npz')
    jsma_x = data['training_jsma_x']


    clean_x_u = training_x[i].reshape(28, 28)
    clean_x_5 = pca_filter_img(training_x, 5)[i]
    clean_x_6 = pca_filter_img(training_x, 6)[i]
    clean_x_7 = pca_filter_img(training_x, 7)[i]
    clean_x_10 = pca_filter_img(training_x, 10)[i]

    # 5, 6, 7, 10
    jsma_x_u = jsma_x[i].reshape(28, 28)
    jsma_x_5 = pca_filter_img(jsma_x, 5)[i]
    jsma_x_6 = pca_filter_img(jsma_x, 6)[i]
    jsma_x_7 = pca_filter_img(jsma_x, 7)[i]
    jsma_x_10 = pca_filter_img(jsma_x, 10)[i]



    plt.figure(1)
    plt.subplot(1, 5, 1)
    plt.imshow(clean_x_u, cmap='gray')
    plt.xlabel('unfiltered')
    plt.title('Clean Data')

    plt.subplot(1, 5, 2)
    plt.imshow(clean_x_5, cmap='gray')
    plt.xlabel('n = 25')

    plt.subplot(1, 5, 3)
    plt.imshow(clean_x_6, cmap='gray')
    plt.xlabel('n = 36')

    plt.subplot(1, 5, 4)
    plt.imshow(clean_x_7, cmap='gray')
    plt.xlabel('n = 49')

    plt.subplot(1, 5, 5)
    plt.imshow(clean_x_10, cmap='gray')
    plt.xlabel('n = 100')


    plt.figure(2)
    plt.subplot(1, 5, 1)
    plt.imshow(jsma_x_u, cmap='gray')
    plt.xlabel('unfiltered')
    plt.title('Adversarial Data')

    plt.subplot(1, 5, 2)
    plt.imshow(jsma_x_5, cmap='gray')
    plt.xlabel('n = 25')

    plt.subplot(1, 5, 3)
    plt.imshow(jsma_x_6, cmap='gray')
    plt.xlabel('n = 36')

    plt.subplot(1, 5, 4)
    plt.imshow(jsma_x_7, cmap='gray')
    plt.xlabel('n = 49')

    plt.subplot(1, 5, 5)
    plt.imshow(jsma_x_10, cmap='gray')
    plt.xlabel('n = 100')

    plt.show()



def pca_filter_img(X, n_components):
    # expand out to features
    X_flat = X.reshape(-1, 28*28)

    n_components_total = int(n_components * n_components)
    pca_tmp = PCA(n_components = n_components_total)

    s_fit_transform = pca_tmp.fit_transform(X_flat)
    # inverse back to original planes
    X_inverse_flat = pca_tmp.inverse_transform(s_fit_transform)

    # back to X_filter
    X_filter = X_inverse_flat.reshape(-1, 28, 28)
    return X_filter

if __name__ == '__main__':
    # plot_jsma_sample()
    # plot_pca_sample()

    test()
    pass