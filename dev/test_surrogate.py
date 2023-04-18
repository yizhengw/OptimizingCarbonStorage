import torch
from gas_saturation_injection_period import *

file_dir = '/home/jmern/Storage/CCS/data/' # change this to your dir
model = torch.load(file_dir + 'sg_inj_model.pt')

sg_input = torch.load(file_dir + 'sg_input_lite.pt')
sg_output = torch.load(file_dir + 'sg_output_lite.pt')
sg_mass_output = torch.load(file_dir + 'sg_mass_output_lite.pt')

device = torch.device('cuda:0')
eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(sg_input[:5, ...], sg_output[:5, ...]),
                                          batch_size=1, shuffle=False)

counter = 0
with torch.no_grad():
    for xx, yy in eval_loader:
        xx = xx.to(device)
        # yy = yy.to(device)
        import pdb; pdb.set_trace()
        sg_pred, mass_pred = model(xx)
        x_plot = xx.cpu().detach().numpy()
        y_plot = yy.cpu().detach().numpy()
        pred_plot = sg_pred.cpu().detach().numpy()  # Full field prediction
        mass_plot = mass_pred.cpu().detach().numpy()
        mass_pred_plot = sg_mass_output.cpu().detach().numpy()  # mass prediction

        plt.figure(figsize=(15, 5))
        plt.jet()
        for it, t in enumerate([0, 5, 10, 15, 20, 25]):
            plt.subplot(3, 6, it + 1)
            plt.title('inj loc')
            plt.imshow(x_plot[0, :, :, 0, t, 1])
            plt.colorbar(fraction=0.02)

            plt.subplot(3, 6, it + 7)
            plt.title('true')
            plt.imshow(y_plot[0, :, :, -1, t, 0])
            plt.colorbar(fraction=0.02)

            plt.subplot(3, 6, it + 13)
            plt.title('pred')
            plt.imshow(pred_plot[0, :, :, -1, t])
            plt.clim([np.min(y_plot[0, :, :, -1, t, 0]), np.max(y_plot[0, :, :, -1, t, 0])])
            plt.colorbar(fraction=0.02)

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 3))
        plt.plot(mass_plot.transpose(), '--')
        plt.plot(mass_pred_plot[counter, :, :].transpose(), '-')
        plt.xlabel('time')
        plt.ylabel('mass')
        counter += 1
        plt.show()
