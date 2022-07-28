import os
import shlex
import logging
import paramiko

from collections.abc import MutableMapping

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.organizer.cfg_client import env_name, root_path, fs_version

logger = logging.getLogger('federatedscope')
logger.setLevel(logging.INFO)


class SSHManager(object):
    def __init__(self, ip, user, psw, ssh_port=22):
        self.ip, self.user, self.psw = ip, user, psw
        self.ssh_port = ssh_port
        self.ssh, self.trans = None, None
        self.setup_fs()
        self.tasks = set()

    def _connect(self):
        self.trans = paramiko.Transport((self.ip, self.ssh_port))
        self.trans.connect(username=self.user, password=self.psw)
        self.ssh = paramiko.SSHClient()
        self.ssh._transport = self.trans
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def _disconnect(self):
        self.trans.close()

    def _exec_cmd(self, command):
        if self.trans is None or self.ssh is None:
            self._connect()
        command = f'source ~/.bashrc; cd ~; {command}'
        _, stdout, stderr = self.ssh.exec_command(command)
        stdout = stdout.read().decode('ascii').strip("\n")
        stderr = stderr.read().decode('ascii').strip("\n")
        return stdout, stderr

    def _exec_python_cmd(self, command):
        if self.trans is None or self.ssh is None:
            self._connect()
        command = f'source ~/.bashrc; conda activate {env_name}; ' \
                  f'cd ~/{root_path}/FederatedScope; {command}'
        _, stdout, stderr = self.ssh.exec_command(command)
        stdout = stdout.read().decode('ascii').strip("\n")
        stderr = stderr.read().decode('ascii').strip("\n")
        return stdout, stderr

    def _check_conda(self):
        """
            Check and install conda env.
        """
        # Check conda
        conda, _ = self._exec_cmd('which conda')
        if conda is None:
            logger.exception('No conda, please install conda first.')
            # TODO: Install conda here
            return False

        # Check env & FS
        output, err = self._exec_cmd(f'conda activate {env_name}; '
                                     f'python -c "import federatedscope; '
                                     f'print(federatedscope.__version__)"')
        if err:
            logger.error(err)
            # TODO: Install FS env here
            return False
        elif output != fs_version:
            logger.warning(f'The installed FS version is {output}, however'
                           f' {fs_version} is required.')
        logger.info(f'Conda environment found, named {env_name}.')
        return True

    def _check_source(self):
        """
            Check and download FS repo.
        """
        fs_path = os.path.join(root_path, 'FederatedScope', '.git')
        output, _ = self._exec_cmd(f'[ -d {fs_path} ] && echo "Found" || '
                                   f'echo "Not found"')
        if output == 'Not found':
            # TODO: git clone here
            logger.exception(f'Repo not find in {fs_path}.')
            return False
        logger.info(f'FS repo Found in {root_path}.')
        return True

    def _check_task_status(self):
        """
            Check task status.
        """
        pass

    def setup_fs(self):
        logger.info("Checking environment, please wait...")
        if not self._check_conda():
            raise Exception('The environment is not configured properly.')
        if not self._check_source():
            raise Exception('The FS repo is not configured properly.')

    def launch_task(self, command):
        self._check_task_status()
        stdout, _ = self._exec_python_cmd(f'nohup python '
                                          f'federatedscope/main.py {command} '
                                          f'> /dev/null 2>&1 & echo $!')
        self.tasks.add(stdout)
        return stdout


def anonymize(info, mask):
    for key, value in info.items():
        if key == mask:
            info[key] = "******"
        elif isinstance(value, dict):
            anonymize(info[key], mask)
    return info


def args2yaml(args):
    init_cfg = global_cfg.clone()
    args = parse_args(shlex.split(args))
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    _, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)
    init_cfg.freeze(inform=False, save=False)
    init_cfg.defrost()
    return init_cfg


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def config2cmdargs(config):
    '''
    Arguments:
        config (dict): key is cfg node name, value is the specified value.
    Returns:
        results (list): cmd args
    '''

    results = []
    for key, value in config.items():
        if value:
            results.append(key)
            results.append(value)
    return results
