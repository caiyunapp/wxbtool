# -*- coding: utf-8 -*-


def norm_t2m(t2m):
    return (t2m - 278.5193277994792) / 21.219592501509624


def norm_t850(t850):
    return (t850 - 274.57741292317706) / 15.572175300640202


def norm_tau(tau):
    return (tau - 6048.8221254006414) / 3096.4446045099244


def norm_tisr(tisr):
    return (tisr - 1074511.0673076923) / 1439848.7984975462


def norm_tcc(tcc):
    return (tcc - 0.6740332964139107) / 0.3626919709448507


def norm_z500(z500):
    return (z500 - 54117.3952323718) / 3353.5545664452306


def norm_z1000(z1000):
    return (z1000 - 736.8600307366787) / 1072.7004633440063


def denorm_t2m(t2m):
    return t2m * 21.219592501509624 + 278.5193277994792


def denorm_t850(t850):
    return t850 * 15.572175300640202 + 274.57741292317706


def denorm_tau(tau):
    return tau * 3096.4446045099244 + 6048.8221254006414


def denorm_tisr(tisr):
    return tisr * 1439848.7984975462 + 1074511.0673076923


def denorm_tcc(tcc):
    return tcc * 0.3626919709448507 + 0.6740332964139107


def denorm_z500(z500):
    return z500 * 3353.5545664452306 + 54117.3952323718


def denorm_z1000(z1000):
    return z1000 * 1072.7004633440063 + 736.8600307366787


normalizors = {
    't2m': norm_t2m,
    't850': norm_t850,
    'tau': norm_tau,
    'tisr': norm_tisr,
    'tcc': norm_tcc,
    'z500': norm_z500,
    'z1000': norm_z1000,
}


denormalizors = {
    't2m': denorm_t2m,
    't850': denorm_t850,
    'tau': denorm_tau,
    'tisr': denorm_tisr,
    'tcc': denorm_tcc,
    'z500': denorm_z500,
    'z1000': denorm_z1000,
}
