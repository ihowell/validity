from .lid import load_best_lid, load_lid, LIDDetector
from .llr import load_best_llr, LikelihoodRatioDetector
from .mahalanobis import MahalanobisDetector, load_best_mahalanobis_adv, load_best_mahalanobis_ood, load_mahalanobis_ood, load_mahalanobis_adv
from .odin import ODINDetector, load_best_odin, load_odin


def load_detectors(in_ds_name, out_ds_name, cls_type, adv_attack, adv_step=True):
    # Load detectors
    llr_ood = load_best_llr(in_ds_name, out_ds_name)

    lid_adv = load_best_lid(cls_type, in_ds_name, adv_attack)
    if adv_step:
        odin_ood = load_best_odin(cls_type, in_ds_name, out_ds_name)
        mahalanobis_ood = load_best_mahalanobis_ood(cls_type, in_ds_name, out_ds_name)

        mahalanobis_adv = load_best_mahalanobis_adv(cls_type, in_ds_name, adv_attack)
    else:
        odin_ood = load_odin(cls_type, in_ds_name, out_ds_name, 0, 1000.)
        mahalanobis_ood = load_mahalanobis_ood(cls_type, in_ds_name, out_ds_name, 0)

        mahalanobis_adv = load_mahalanobis_adv(cls_type, in_ds_name, adv_attack, 0)

    ood_detectors = [
        (f'ODIN OOD {out_ds_name}', odin_ood),
        (f'Mahalanobis OOD {out_ds_name}', mahalanobis_ood),
        (f'LLR OOD {out_ds_name}', llr_ood),
    ]
    adv_detectors = [
        (f'LID Adv {adv_attack}', lid_adv),
        (f'Mahalanobis Adv {adv_attack}', mahalanobis_adv),
    ]
    return ood_detectors, adv_detectors
