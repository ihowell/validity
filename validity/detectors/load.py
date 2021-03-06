from .lid import load_best_lid, load_lid, LIDDetector
from .llr import load_llr, LikelihoodRatioDetector
from .mahalanobis import MahalanobisDetector, load_best_mahalanobis_adv, load_best_mahalanobis_ood, load_mahalanobis_ood, load_mahalanobis_adv
from .odin import ODINDetector, load_best_odin, load_odin


def load_detectors(in_ds_name, out_ds_name, cls_type, adv_attack, adv_step=True, id=None):
    # Load detectors
    # llr_ood = load_best_llr(in_ds_name, out_ds_name)
    llr_ood = load_llr(in_ds_name, out_ds_name, 0.3, id=id)

    lid_adv = load_best_lid(cls_type, in_ds_name, adv_attack, id=id)
    if adv_step:
        odin_ood = load_best_odin(cls_type, in_ds_name, out_ds_name, id=id)
        mahalanobis_ood = load_best_mahalanobis_ood(cls_type,
                                                    in_ds_name,
                                                    out_ds_name,
                                                    classifier_id=id)

        mahalanobis_adv = load_best_mahalanobis_adv(cls_type,
                                                    in_ds_name,
                                                    adv_attack,
                                                    classifier_id=id)
    else:
        odin_ood = load_odin(cls_type, in_ds_name, out_ds_name, 0, 1000., id=id)
        mahalanobis_ood = load_mahalanobis_ood(cls_type,
                                               in_ds_name,
                                               out_ds_name,
                                               0,
                                               classifier_id=id)

        mahalanobis_adv = load_mahalanobis_adv(cls_type,
                                               in_ds_name,
                                               adv_attack,
                                               0,
                                               classifier_id=id)

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


def load_ood_detectors(cls_type, in_ds_name, out_ds_name, classifier_id=None):
    llr_ood = load_llr(in_ds_name, out_ds_name, 0.3, id=classifier_id)
    odin_ood = load_best_odin(cls_type, in_ds_name, out_ds_name, id=classifier_id)
    mahalanobis_ood = load_best_mahalanobis_ood(cls_type,
                                                in_ds_name,
                                                out_ds_name,
                                                classifier_id=classifier_id)

    ood_detectors = [
        (f'ODIN OOD {out_ds_name}', odin_ood),
        (f'Mahalanobis OOD {out_ds_name}', mahalanobis_ood),
        (f'LLR OOD {out_ds_name}', llr_ood),
    ]
    return ood_detectors


def load_adv_detectors(cls_type, ds_name, adv_attack, classifier_id=None):
    lid_adv = load_best_lid(cls_type, ds_name, adv_attack, id=classifier_id)
    mahalanobis_adv = load_best_mahalanobis_adv(cls_type,
                                                ds_name,
                                                adv_attack,
                                                classifier_id=classifier_id)

    adv_detectors = [
        (f'LID Adv {adv_attack}', lid_adv),
        (f'Mahalanobis Adv {adv_attack}', mahalanobis_adv),
    ]
    return adv_detectors
