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
def plot():
    clean = [0.976000022888,0.980000019073,0.982000017166,
             0.979000020027,0.977000021935,0.978000020981,
             0.979000020027, 0.977000021935
             ]
    jsma = [0.148999993503,0.150999994576,0.165999997407,
            0.179999993742,0.245999993384,0.24699999392,
            0.253999990225,0.283999986947
            ]
    en = [0.0439999978989,0.0419999992475,0.0349999988452,
          0.0439999973401,0.0509999984875,0.0489999981597,
          0.0549999976531,0.0449999988079
          ]

    x = np.arange(8)
    plt.plot(x, clean, 'r', label='clean data accuracy')
    plt.plot(x, jsma, 'b', label='JSMA data accuracy')
    plt.plot(x, en, 'g', label='EN data accuracy')
    plt.legend()
    # x, jsma, 'bs', x, en, 'g^'
    plt.title('Testing Accuracy vs Epochs')
    plt.show()




if __name__ == '__main__':
    plot()
