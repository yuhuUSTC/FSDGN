import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from basicsr.utils import imwrite, tensor2img
from torchvision import utils
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.models.crop_validation import forward_crop

#from tensorboardX import SummaryWriter
#logger1 = SummaryWriter(log_dir="D:\VideoDehazing\BasicSR\datasets")


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.prenet = build_network(opt['network_pre'])
        self.prenet = self.model_to_device(self.prenet)
        #self.print_network(self.net_g)
        #self.print_network(self.prenet)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.prenet, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        #self.prenet.eval()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('contrastive_opt'):
            self.cri_contrastive = build_loss(train_opt['contrastive_opt']).to(self.device)
        else:
            self.cri_contrastive = None

        if train_opt.get('l1_opt'):
            self.cri_l1 = build_loss(train_opt['l1_opt']).to(self.device)
        else:
            self.cri_l1 = None

        if train_opt.get('lab_opt'):
            self.cri_lab = build_loss(train_opt['lab_opt']).to(self.device)
        else:
            self.cri_lab = None

        if train_opt.get('gradient_opt'):
            self.cri_gradient = build_loss(train_opt['gradient_opt']).to(self.device)
        else:
            self.cri_gradient = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color  = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('pre_opt'):
            self.cri_pre = build_loss(train_opt['pre_opt']).to(self.device)
        else:
            self.cri_pre = None

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self._, self.pre, self.T, self.A, self.I = self.prenet(self)
        #self.output = self.net_g(self)
        self.output = self.net_g(self)
        #logger1.add_graph(self.net_g, input_to_model=self.lq, verbose=False)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_lab:
            l_lab = self.cri_lab(self.output, self.gt)
            l_total += l_lab
            loss_dict['l_lab'] = l_lab

        if self.cri_gradient:
            l_gradient = self.cri_gradient(self.output, self.gt)
            l_total += l_gradient
            loss_dict['l_gradient'] = l_gradient

        if self.cri_pre:
            l_pre = self.cri_pre(self.mask, self.pre, self.gt)
            l_total += 0.5*l_pre
            loss_dict['l_pre'] = l_pre

        if self.cri_edge:
            l_edge = self.cri_edge(self.output, self.gt)
            l_total += 0.05*l_edge
            loss_dict['l_edge'] = l_edge

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total += 0.05*l_ssim
            loss_dict['l_ssim'] = l_ssim

        # contrastive loss
        if self.cri_contrastive:
            l_cont = self.cri_contrastive(self.output, self.gt, self.lq[:, 1, :, :, :])
            l_total += l_cont
            loss_dict['l_cont'] = l_cont

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt[:, 1, :, :, :])
            if l_percep is not None:
                l_total += 0.5*l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += 0.5*l_style
                loss_dict['l_style'] = l_style

        if self.cri_l1:
            l_l1 = self.cri_l1(self.output, self.gt[:, 1, :, :, :])
            l_total += l_l1
            loss_dict['l_l1'] = l_l1

        if self.cri_color:
            l_color = self.cri_color(self.output)
            l_total += l_color
            loss_dict['l_color'] = l_color

        #l_total.backward()
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # 添加的第一条日志：损失函数-全局迭代次数
        #logger1.add_scalar("train loss", l_pix.item(), global_step=current_iter)

        #img = utils.make_grid(self.lq[:, 1, :, :, :], normalize=True, scale_each=True, nrow=1)
        #logger1.add_image("train image sample", img, global_step=current_iter)

        # 添加第三条日志：网络中的参数分布直方图
        #if current_iter % 1000 == 0:
        #    for name, param in self.net_g.named_parameters():
        #        logger1.add_histogram(name, param.data.cpu().numpy(), global_step=current_iter)



    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if self.opt['crop']:
                # if self.opt['train'].get('flow_opt'):
                #     self.output = forward_crop(self.lq, self.net_g, flow_opt=self.opt['train'].get('flow_opt'))
                # else:
                if 'train' in self.opt['datasets']:
                    lq_size = self.opt['datasets']['train']['gt_size']
                if 'test' in self.opt['datasets']:
                    lq_size = self.opt['datasets']['test']['lq_size']
                overlap = lq_size // 2   # TODO
                self.output = forward_crop(self.lq, self.net_g, lq_size=lq_size, overlap=overlap)
            else:
                # if self.opt['train'].get('flow_opt'):
                #     self.output, _ = self.net_g(self.lq)
                # else:
                #self.pre = self.prenet(self)
                self._, self.pre, self.T, self.A, self.I = self.prenet(self)
                self.output = self.net_g(self)
                a = self.pre[0, 1, :, :, :]
                a = tensor2img(a)
                b = self.T[0, 1, :, :, :]
                b = tensor2img(b)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
                cv2.imshow('image', a)
                cv2.imshow('image1', b)
                cv2.waitKey(0)
                #self.output = self.output[:, 1, :, :, :]
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
