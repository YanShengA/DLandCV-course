import torch
from .base_model import BaseModel
from . import networks
import torchvision.models as models
from math import exp
import torch.nn.functional as F
# === SSIM Helper Classes Start ===
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean() # 返回 1 - SSIM 作为 Loss
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)
# === SSIM Helper Classes End ===

class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        if hasattr(opt, 'use_vgg') and opt.use_vgg:
            self.loss_names.append("G_VGG")
        if hasattr(opt, 'use_fm') and opt.use_fm:
            self.loss_names.append("G_FM")
        if opt.use_ssim:
            self.loss_names.append('G_SSIM') # 如果开启，增加 G_SSIM 显示
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.use_resize_conv, opt.use_dilated_conv, opt.use_large_kernel, opt.use_attention)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.use_spectral_norm)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
            self.criterionL1 = torch.nn.L1Loss()
            if opt.use_ssim:
                self.criterionSSIM = SSIMLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # 1. 生成 Discriminator 需要的输入 (Conditional GAN 需要拼接 A 和 B)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        # 2. 计算基础 GAN Loss (G 骗 D)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # 3. 计算 L1 Loss (像素级对齐)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G_SSIM = 0
        if self.opt.use_ssim:
            # 权重建议设为 10 (因为 L1 通常是 100，SSIM 值较小，稍微放大一点权值)
            # 或者按照常用配置：L1 * 0.8 + SSIM * 0.2 (这里我们简单叠加)
            self.loss_G_SSIM = self.criterionSSIM(self.fake_B, self.real_B) * 10 
        
        # 总 Loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SSIM

        # ============================================================
        # 4. [Exp D/E] VGG Perceptual Loss (感知损失)
        # ============================================================
        # 检查是否启用了 use_vgg 参数
        if hasattr(self.opt, 'use_vgg') and self.opt.use_vgg:
            # 懒加载 VGG，只在第一次用到时加载，节省显存
            if not hasattr(self, 'vgg'):
                # 加载 VGG19 预训练模型，取前 18 层 (relu3_x 附近)
                self.vgg = models.vgg19(pretrained=True).features[:18].to(self.device).eval()
                # 冻结参数，绝对不要更新 VGG
                for p in self.vgg.parameters():
                    p.requires_grad = False
                self.criterionVGG = torch.nn.L1Loss()
            
            # 计算 VGG 特征距离
            vgg_fake = self.vgg(self.fake_B)
            vgg_real = self.vgg(self.real_B)
            
            # 获取权重，默认为 10.0
            w_vgg = getattr(self.opt, 'vgg_weight', 10.0)
            self.loss_G_VGG = self.criterionVGG(vgg_fake, vgg_real) * w_vgg
            
            # 加到总 Loss
            self.loss_G += self.loss_G_VGG

        # ============================================================
        # 5. [Exp C/E] Feature Matching Loss (特征匹配损失)
        # ============================================================
        # 检查是否启用了 use_fm 参数
        if hasattr(self.opt, 'use_fm') and self.opt.use_fm:
            # FM Loss 需要比较 D 对“真图”和“假图”的中间层反应
            # 我们需要计算 pred_real (注意：必须 detach，不更新 D)
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            
            w_fm = getattr(self.opt, 'fm_weight', 10.0)
            
            # 兼容性处理：Discriminator 可能返回列表 (Multiscale) 或 Tensor
            loss_fm = 0
            if isinstance(pred_fake, list):
                # 如果是多尺度 D，把每一层的差异加起来
                for i in range(len(pred_fake)):
                    # .detach() 很重要！我们不希望 G 的梯度传导去改变 D 的权重
                    loss_fm += torch.mean(torch.abs(pred_fake[i] - pred_real[i].detach()))
            else:
                loss_fm = torch.mean(torch.abs(pred_fake - pred_real.detach()))
            
            self.loss_G_FM = loss_fm * w_fm
            
            # 加到总 Loss
            self.loss_G += self.loss_G_FM

        # ============================================================
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
