"""
algorithm
"""

import torch.optim as optim
import torch.nn as nn
from propagation_ASM import *
from pytorch_msssim import ssim as ssim_func
import torch.nn.functional as func

# 2. SGD
def stochastic_gradient_descent(init_phase, target_amp, num_iters, prop_dist, wavelength, feature_size,
                                roi_res=None, phase_path=None, prop_model='ASM', propagator=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0, citl=False, camera_prop=None,
                                writer=None, dtype=torch.float32, precomputed_H_exp=None,target_dep=None,bit=None):

    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: A Pytorch complex64 tensor, pre-computed kernel shape of (1,1,2H,2W) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    admm_opt=None
    # admm_opt = {'num_iters_inner': 50,'rho': 0.01,'alpha': 1.0,'gamma': 0.1,'varying-penalty': True,'mu': 10.0,'tau_incr': 2.0,'tau_decr': 2.0}
    num_iters_admm_inner = 1 if admm_opt is None else admm_opt['num_iters_inner']

    # crop target roi
    target_amp = func.interpolate(target_amp, scale_factor=2, mode='nearest')
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)

    bit= bit
    sign = utils.LBSign.apply
    relu = utils.LBRelu.apply

    # if admm_opt is not None:
    u = torch.zeros(1, 1, *roi_res).to(device)
    z = torch.zeros(1, 1, *roi_res).to(device)

    weight_decay0 = 0.01  #bit=8，ref=-0.0091时，wet为0.2时较好
    TV_decay0=0.1
    RefWaveTheta = 0
    offset = -10000 # best:-20
    mode=0 #mode=0,优化相位,mode=1,优化振幅
    timeN=init_phase.size(0)
    slm_phaseN=torch.zeros_like(init_phase)

    for timen in range(timeN):
        print('******************','timen:',timen)
        slm_phase = init_phase[timen,:,:,:].unsqueeze(0).requires_grad_(True)
        optvars = [{'params': slm_phase}]
        optimizer = optim.Adam(optvars, lr=lr)
        loss_vals = []
        loss_vals_quantized = []
        best_loss = 10.
        best_iter = 0
        if admm_opt is not None:
            rho=admm_opt['rho']

        # run the iterative algorithm
        for k in range(num_iters):
            print(k)
            for t_inner in range(num_iters_admm_inner):
                optimizer.zero_grad()
                # forward propagation from the SLM plane to the target plane
                slm_phaseBit = utils.BitArraySGD(sign, relu, slm_phase, bit, k, num_iters,mode,dtype,device)
                real, imag = utils.polar_to_rect_mode(slm_phaseBit,mode,bit)
                slm_field = torch.complex(real, imag)
                # slm_field = utils.FieldAddRefWave(slm_phaseBit, RefWaveTheta, wavelength, -prop_dist, dtype)
                # slm_field = utils.FieldAddRefWave(slm_field, RefWaveTheta, wavelength, -prop_dist,dtype)
                recon_field = utils.propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size,
                                            prop_model, dtype, precomputed_H_exp,offset,target_dep)

                # crop roi
                recon_field  = utils.crop_image(recon_field , target_shape=roi_res, stacked_complex=False)

                # get amplitude
                recon_amp = recon_field.abs()
                out_amp = recon_amp

                with torch.no_grad():
                    s = (out_amp * target_amp).mean() / \
                        (out_amp ** 2).mean()  # scale minimizing MSE btw recon and

    #             # calculate loss and backprop
                mse_loss = loss(s * out_amp, target_amp)
                psnr_loss = utils.psnr_func(target_amp, s * out_amp, data_range=1)
                ssim_loss = 1 - ssim_func(target_amp, s * out_amp, data_range=2)

                lossValue = mse_loss
                # lossValue = mse_loss+ssim_loss*0.5     #flower较好时取0.5,outlineflower较好时取0.005左右
                # lossValue = mse_loss + ssim_loss * 0.5+ phase_loss
                # lossValue = mse_loss + phase_loss

                # second loss term if ADMM
                # recon_phase = utils.phaseCal(recon_field)
                # loss_prior = loss(utils.laplacian(recon_phase), (z - u))
                # L1loss = nn.L1Loss()
                # loss_prior = L1loss(utils.laplacian(recon_phase), (z - u))
                # recon_phase = utils.unwrap(recon_phase)
                # phaseEdge = utils.functional_conv2d(recon_phase,dtype,recon_phase.device.type)
                # loss_prior=torch.norm(phaseEdge,p=1)/(recon_phase.size(2)*recon_phase.size(3))
                # lossValue = lossValue + 0.01 * loss_prior
                # if admm_opt is not                                      None:
                #     # augmented lagrangian
                #     lossValue = lossValue + admm_opt['rho'] * loss_prior

                print('mse:', mse_loss,'psnr:', psnr_loss, 'ssim:', 1 - ssim_loss)

                lossValue.backward()
                optimizer.step()

                ## ADMM steps
                if admm_opt is not None:
                    with torch.no_grad():
                        reg_norm = utils.laplacian(recon_phase).detach()
                        Ax = admm_opt['alpha'] * reg_norm + (1 - admm_opt['alpha']) * z  # over-relaxation
                        z = utils.soft_thresholding(u + Ax, admm_opt['gamma'] / (rho + 1e-10))
                        u = u + Ax - z

                        # varying penalty (rho)
                        if admm_opt['varying-penalty']:
                            if k == 0:
                                z_prev = z

                            r_k = ((reg_norm - z).detach() ** 2).mean()  # primal residual
                            s_k = ((rho * utils.laplacian(z_prev - z).detach()) ** 2).mean()  # dual residual

                            if r_k > admm_opt['mu'] * s_k:
                                rho = admm_opt['tau_incr'] * rho
                                u /= admm_opt['tau_incr']
                            elif s_k > admm_opt['mu'] * r_k:
                                rho /= admm_opt['tau_decr']
                                u *= admm_opt['tau_decr']
                            z_prev = z

                # with torch.no_grad():
                #     if lossValue < best_loss:
                #         best_phase = slm_phaseBit
                #         best_loss = lossValue
                #         best_amp = s * recon_amp
                #         best_iter = k + 1

        # slm_phaseBit=best_phase
        slm_phaseN[timen:timen + 1] = slm_phaseBit
        utils.resultshow(slm_phaseBit, mode, bit, RefWaveTheta, wavelength, prop_dist, dtype, propagator, feature_size,prop_model, precomputed_H_exp, offset, roi_res, s, target_amp)

    return slm_phaseN