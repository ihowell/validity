import torch

from .models.GAN import StyleGAN

def _construct_stylegan_from_args(args):
    stylegan = StyleGAN(structure=args.structure,
                        conditional=args.conditional,
                        n_classes=args.n_classes,
                        resolution=args.dataset.resolution,
                        num_channels=args.dataset.channels,
                        latent_size=args.model.gen.latent_size,
                        g_args=args.model.gen,
                        d_args=args.model.dis,
                        g_opt_args=args.model.g_optim,
                        d_opt_args=args.model.d_optim,
                        loss=args.loss,
                        drift=args.drift,
                        d_repeats=args.d_repeats,
                        use_ema=args.use_ema,
                        ema_decay=args.ema_decay)
    return stylegan

def _load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def load_stylegan(checkpoint_path, batch_size=0):
    pretrained_dict = torch.load(checkpoint_path)
    args = pretrained_dict['opt']
    assert(args is not None)
    model = _construct_stylegan_from_args(args)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
