from validity.adv_dataset import construct_dataset as construct_adv_dataset
from validity.classifiers import load_cls
from validity.classifiers.mnist import train_network as train_mnist_cls
from validity.detectors.density import train_density_adv
from validity.detectors.lid import train_multiple_lid_adv
from validity.detectors.llr import train_llr_ood
from validity.detectors.mahalanobis import train_multiple_mahalanobis_adv, \
    train_multiple_mahalanobis_ood
from validity.generators.mnist_vae import train as train_mnist_vae, \
    get_save_path as get_mnist_vae_path
from validity.generators.wgan_gp import train as train_wgan_gp


def run_experiment():
    mnist_cls_path = 'models/cls_mnist_mnist.pt'
    adv_attacks = ['fgsm', 'bim', 'cwl2']

    # Train classifiers
    train_mnist_cls()

    # Train generative functions
    train_mnist_vae(beta=20.)
    train_mnist_vae(beta=20., mutation_rate=0.3)
    train_wgan_gp('mnist')

    # Create adversarial datasets
    construct_adv_dataset('mnist', 'fgsm', 'mnist', mnist_cls_path)
    construct_adv_dataset('mnist', 'bim', 'mnist', mnist_cls_path)
    construct_adv_dataset('mnist', 'cwl2', 'mnist', mnist_cls_path)

    # Train OOD detectors
    train_llr_ood('mnist', 'fmnist', 'mnist_vae', get_mnist_vae_path(beta=20.),
                  get_mnist_vae_path(beta=20., mutation_rate=0.3), 0.3)
    train_multiple_mahalanobis_ood('mnist', 'fmnist', 'mnist', mnist_cls_path)

    # Train ADV detectors
    for adv_attack in adv_attacks:
        train_density_adv('mnist', 'mnist', mnist_cls_path, adv_attack)
        train_multiple_lid_adv('mnist', 'mnist', mnist_cls_path, adv_attack)
        train_multiple_mahalanobis_adv('mnist', 'mnist', mnist_cls_path, adv_attack)

    # Evaluate detector joint distribution and error
