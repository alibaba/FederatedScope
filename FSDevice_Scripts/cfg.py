from yacs.config import CfgNode as CN


def get_fl_cfg():
    cfg = CN()

    cfg.workdir = ""  # workdir for the current project, storage media files

    cfg.sampler = "fix"

    cfg.server = CN()
    cfg.server.host = ""
    cfg.server.port = -1
    cfg.server.workdir = ""
    cfg.server.user = ""
    cfg.server.passwd = ""

    cfg.machine = CN()
    cfg.machine.host_path = "./machines.txt"
    cfg.machine.workdir = ""
    cfg.machine.user = ""
    cfg.machine.passwd = ""

    cfg.fl = CN()
    cfg.fl.n_client = -1
    cfg.fl.cfg_path = ""
    cfg.fl.apk_path = ""
    cfg.fl.apk_name = ""
    cfg.fl.avd_path = ""
    cfg.fl.server_path = ""

    return cfg

def get_lt_cfg():
    cfg = CN()

    cfg.server = CN()
    cfg.server.workdir = ""
    cfg.server.host = ""
    cfg.server.user = ""
    cfg.server.passwd = ""

    cfg.lt = CN()
    cfg.lt.apk_path = ""
    cfg.lt.avd_path = ""

    return cfg
