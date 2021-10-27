import torch

from torch.autograd import Variable

from model import AutoEncoder

import utils

if __name__ == '__main__':
    checkpoint = torch.load(
        '/mnt/d/research/nvae_models/cifar10/qualitative/checkpoint.pt')
    args = checkpoint['args']
    args.num_mixture_dec = 10
    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    dummy_input = Variable(torch.randn([1, 3, 32, 32])).cuda()
    torch.onnx.export(model,
                      dummy_input,
                      'cifar10_nvae.onnx',
                      opset_version=14)
