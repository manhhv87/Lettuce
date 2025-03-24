import numpy as np
import matplotlib.pyplot as plt
from metrics.nmse import nmse
from metrics.nrmse import nrmse
from sklearn.metrics import r2_score
from dataset.data_preprocess import *
from dataset.read_label import *

# read data
train_loss_fw = np.load('result/Model_11/train_loss_fw.npy')
test_loss_fw = np.load('result/Model_11/test_loss_fw.npy')
train_loss_dw = np.load('result/Model_11/train_loss_dw.npy')
test_loss_dw = np.load('result/Model_11/test_loss_dw.npy')
train_loss_dia = np.load('result/Model_11/train_loss_dia.npy')
test_loss_dia = np.load('result/Model_11/test_loss_dia.npy')
train_loss_area = np.load('result/Model_11/train_loss_area.npy')
test_loss_area = np.load('result/Model_11/test_loss_area.npy')

train_loss_var = np.load('result/Model_12/train_loss.npy')
test_loss_var = np.load('result/Model_12/val_loss.npy')
train_acc_var = np.load('result/Model_12/train_acc.npy')
test_acc_var = np.load('result/Model_12/val_acc.npy')

train_loss_h = np.load('result/Model_13/train_loss_h.npy')
test_loss_h = np.load('result/Model_13/test_loss_h.npy')

# Plot results
plt.subplots_adjust(wspace=0.5, hspace=0.8)  # Adjust sub-image spacing

plt.subplot(2, 4, 1)
plt.title('The loss curve of FW')
plt.xlabel("Epochs")
plt.ylabel("MSE(gram$^2$)")
plt.plot(np.arange(50), train_loss_fw, c='blue', label='train_fw')
plt.plot(np.arange(50), test_loss_fw, c='orange', label='test_fw')
plt.legend()

plt.subplot(2, 4, 2)
plt.title('The loss curve of DW')
plt.xlabel("Epochs")
plt.ylabel("MSE(gram$^2$)")
plt.plot(np.arange(50), train_loss_dw, c='blue', label='train_dw')
plt.plot(np.arange(50), test_loss_dw, c='orange', label='test_dw')
plt.legend()

plt.subplot(2, 4, 3)
plt.title('The loss curve of Diameter')
plt.xlabel("Epochs")
plt.ylabel("MSE(cm$^2$)")
plt.plot(np.arange(50), train_loss_dia, c='blue', label='train_dia')
plt.plot(np.arange(50), test_loss_dia, c='orange', label='test_dia')
plt.legend()

plt.subplot(2, 4, 4)
plt.title('The loss curve of LA')
plt.xlabel("Epochs")
plt.ylabel("MSE(cm$^4$)")
plt.plot(np.arange(50), train_loss_area, c='blue', label='train_area')
plt.plot(np.arange(50), test_loss_area, c='orange', label='test_area')
plt.legend()

plt.subplot(2, 4, 5)
plt.title('The loss curve of Varieties')
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy")
plt.plot(np.arange(50), train_loss_area, c='blue', label='train_loss')
plt.plot(np.arange(50), test_loss_area, c='orange', label='test_loss')
plt.legend()

plt.subplot(2, 4, 6)
plt.title('The accuracy curve of Varieties')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(np.arange(50), train_acc_var, c='blue', label='train_acc')
plt.plot(np.arange(50), test_acc_var, c='orange', label='test_acc')
plt.legend()

plt.subplot(2, 4, 7)
plt.title('The loss curve of Height')
plt.xlabel("Epochs")
plt.ylabel("MSE(cm$^2$)")
plt.plot(np.arange(50), train_loss_h, c='blue', label='train_loss')
plt.plot(np.arange(50), test_loss_h, c='orange', label='test_loss')
plt.legend()

plt.show()
