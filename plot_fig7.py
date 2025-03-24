import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from metrics.nrmse import nrmse

# read data
pred_fw = np.load('result/eval/pred_fw.npy')
true_fw = np.load('result/eval/true_fw.npy')
pred_dw = np.load('result/eval/pred_dw.npy')
true_dw = np.load('result/eval/true_dw.npy')
pred_h = np.load('result/eval/pred_h.npy')
true_h = np.load('result/eval/true_h.npy')
pred_dia = np.load('result/eval/pred_dia.npy')
true_dia = np.load('result/eval/true_dia.npy')
pred_la = np.load('result/eval/pred_area.npy')
true_la = np.load('result/eval/true_area.npy')

r2_fw, nrmse_fw = r2_score(true_fw, pred_fw), nrmse(true_fw, pred_fw)
r2_dw, nrmse_dw = r2_score(true_dw, pred_dw), nrmse(true_dw, pred_dw)
r2_h, nrmse_h = r2_score(true_h, pred_h), nrmse(true_h, pred_h)
r2_dia, nrmse_dia = r2_score(true_dia, pred_dia), nrmse(true_dia, pred_dia)
r2_la, nrmse_la = r2_score(true_la, pred_la), nrmse(true_la, pred_la)

# Plot results
fig, ax = plt.subplots(2,3)
plt.subplots_adjust(wspace=0.6, hspace=0.6)  # Adjust sub-image spacing

## Plot FW
# Perform linear fit
coefficients_fw = np.polyfit(true_fw, np.squeeze(pred_fw), deg=1)

# Create polynomial function
p_fw = np.poly1d(coefficients_fw)

# Add points to plot
ax[0,0].scatter(true_fw, pred_fw, color='orange', s=20)

# Add line of best fit to plot
ax[0,0].plot(true_fw, p_fw(true_fw), color='red', linewidth=.7)

# Add y=x line
ax[0,0].axline((0, 0), slope=1, linestyle='--', linewidth=1.2)

# Add fitted regession equation to plot
ax[0,0].text(20, 440, 'y = ' + '{:.2f}'.format(coefficients_fw[0]) + 'x' + ' + ' + '{:.2f}'.format(coefficients_fw[1]), size=8)
ax[0,0].text(20, 400, '$r^2$ = {0:.4f}'.format(r2_fw), size=8)
ax[0,0].text(20, 360, 'NRMSE = {0:.2f}%'.format(nrmse_fw), size=8)
ax[0,0].set_xlim(0,500)
ax[0,0].set_ylim(0,500)
ax[0,0].set_xlabel('Measured FW(gram/plant)', fontsize=10)
ax[0,0].set_ylabel('Estimated FW(gram/plant)', fontsize=10)

## Plot DW
# Perform linear fit
coefficients_dw = np.polyfit(true_dw, np.squeeze(pred_dw), deg=1)

# Create polynomial function
p_dw = np.poly1d(coefficients_dw)

# I move the subplot_adjust here before you create ax5
fig.subplots_adjust(right=0.8) 

# Add points to plot
ax[0,1].scatter(true_dw, pred_dw, color='orange', s=20)

# Add line of best fit to plot
ax[0,1].plot(true_dw, p_dw(true_dw), color='red', linewidth=.7)

# Add y=x line
ax[0,1].axline((0, 0), slope=1, linestyle='--', linewidth=1.2)

# Add fitted regession equation to plot
ax[0,1].text(10, 6, 'y = ' + '{:.2f}'.format(coefficients_dw[0]) + 'x' + ' + ' + '{:.2f}'.format(coefficients_dw[1]), size=8)
ax[0,1].text(10, 4, '$r^2$ = {0:.4f}'.format(r2_dw), size=8)
ax[0,1].text(10, 2, 'NRMSE = {0:.2f}%'.format(nrmse_dw), size=8)
ax[0,1].set_xlim(0,22)
ax[0,1].set_ylim(0,22)
ax[0,1].set_xlabel('Measured DW(gram/plant)', fontsize=10)
ax[0,1].set_ylabel('Estimated DW(gram/plant)', fontsize=10)

## Plot Height
# Perform linear fit
coefficients_h = np.polyfit(true_h, np.squeeze(pred_h), deg=1)

# Create polynomial function
p_h = np.poly1d(coefficients_h)

# I move the subplot_adjust here before you create ax5
fig.subplots_adjust(right=0.8) 

# Add points to plot
ax[0,2].scatter(true_h, pred_h, color='orange', s=20)

# Add line of best fit to plot
ax[0,2].plot(true_h, p_h(true_h), color='red', linewidth=.7)

# Add y=x line
ax[0,2].axline((0, 0), slope=1, linestyle='--', linewidth=1.2)

# Add fitted regession equation to plot
ax[0,2].text(2, 23, 'y = ' + '{:.2f}'.format(coefficients_h[0]) + 'x' + ' + ' + '{:.2f}'.format(coefficients_h[1]), size=8)
ax[0,2].text(2, 21, '$r^2$ = {0:.4f}'.format(r2_h), size=8)
ax[0,2].text(2, 19, 'NRMSE = {0:.2f}%'.format(nrmse_h), size=8)
ax[0,2].set_xlim(0,26)
ax[0,2].set_ylim(0,26)
ax[0,2].set_xlabel('Measured Height(cm)', fontsize=10)
ax[0,2].set_ylabel('Estimated Height(cm)', fontsize=10)

## Plot Diameter
# Perform linear fit
coefficients_dia = np.polyfit(true_dia, np.squeeze(pred_dia), deg=1)

# Create polynomial function
p_dia = np.poly1d(coefficients_dia)

# I move the subplot_adjust here before you create ax5
fig.subplots_adjust(right=0.8) 

# Add points to plot
ax[1,0].scatter(true_dia, pred_dia, color='orange', s=20)

# Add line of best fit to plot
ax[1,0].plot(true_dia, p_dia(true_dia), color='red', linewidth=.7)

# Add y=x line
ax[1,0].axline((0, 0), slope=1, linestyle='--', linewidth=1.2)

# Add fitted regession equation to plot
ax[1,0].text(8, 36, 'y = ' + '{:.2f}'.format(coefficients_dia[0]) + 'x' + ' + ' + '{:.2f}'.format(coefficients_dia[1]), size=8)
ax[1,0].text(8, 33, '$r^2$ = {0:.4f}'.format(r2_dia), size=8)
ax[1,0].text(8, 30, 'NRMSE = {0:.2f}%'.format(nrmse_dia), size=8)
ax[1,0].set_xlim(6,40)
ax[1,0].set_ylim(6,40)
ax[1,0].set_xlabel('Measured Diameter(cm)', fontsize=10)
ax[1,0].set_ylabel('Estimated Diameter(cm)', fontsize=10)

## Plot LA
# Perform linear fit
coefficients_la = np.polyfit(true_la, np.squeeze(pred_la), deg=1)

# Create polynomial function
p_la = np.poly1d(coefficients_la)

# I move the subplot_adjust here before you create ax5
fig.subplots_adjust(right=0.8) 

# Add points to plot
ax[1,1].scatter(true_la, pred_la, color='orange', s=20)

# Add line of best fit to plot
ax[1,1].plot(true_la, p_la(true_la), color='red', linewidth=.7)

# Add y=x line
ax[1,1].axline((0, 0), slope=1, linestyle='--', linewidth=1.2)

# Add fitted regession equation to plot
ax[1,1].text(2500, 1600, 'y = ' + '{:.2f}'.format(coefficients_la[0]) + 'x' + ' + ' + '{:.2f}'.format(coefficients_la[1]), size=8)
ax[1,1].text(2500, 1100, '$r^2$ = {0:.4f}'.format(r2_la), size=8)
ax[1,1].text(2500, 600, 'NRMSE = {0:.2f}%'.format(nrmse_la), size=8)
ax[1,1].set_xlim(0,6300)
ax[1,1].set_ylim(0,6300)
ax[1,1].set_xlabel('Measured LA(cm$^2$)', fontsize=10)
ax[1,1].set_ylabel('Estimated LA(cm$^2$)', fontsize=10)

# Hidden subplot(1,2)
ax[-1, -1].axis('off')

plt.show()