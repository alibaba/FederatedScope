import os
import logging
import paramiko

from federatedscope.organizer.cfg_client import env_name, root_path, fs_version

logger = logging.getLogger('federatedscope')
logger.setLevel(logging.INFO)


class SSHManager(object):
    def __init__(self, ip, user, psw, port, ssh_port=22):
        self.ssh_port = ssh_port
        self.ip = ip
        self.user = user
        self.psw = psw
        self.port = port
        self.ssh, self.trans = None, None

    def _connect(self):
        self.trans = paramiko.Transport((self.ip, self.ssh_port))
        self.trans.connect(username=self.user, password=self.psw)
        self.ssh = paramiko.SSHClient()
        self.ssh._transport = self.trans
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def _disconnect(self):
        self.trans.close()

    def exec_cmd(self, command):
        if self.trans is None or self.ssh is None:
            self._connect()
        command = f'source ~/.bashrc; cd ~; {command}'
        _, stdout, stderr = self.ssh.exec_command(command)
        stdout = stdout.read().decode('ascii').strip("\n")
        stderr = stderr.read().decode('ascii').strip("\n")
        return stdout, stderr

    def exec_python_cmd(self, command):
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
        conda, _ = self.exec_cmd('which conda')
        if conda is None:
            logger.exception('No conda, please install conda first.')
            # TODO: Install conda here
            return False

        # Check env & FS
        output, err = self.exec_cmd(f'conda activate {env_name}; '
                                    f'python -c "import federatedscope; print('
                                    f'federatedscope.__version__)"')
        if err:
            logger.error(err)
            # TODO: Install FS env here
            return False
        logger.info('Conda environment found.')
        return True

    def _check_source(self):
        """
            Check and download FS repo.
        """
        fs_path = os.path.join(root_path, 'FederatedScope', '.git')
        output, _ = self.exec_cmd(f'[ -d {fs_path} ] && echo "Found" || '
                                  f'echo "Not found"')
        if output == 'Not found':
            # TODO: git clone here
            logger.exception(f'Repo not find in {fs_path}.')
            return False
        logger.info(f'FS repo Found in {root_path}.')
        return True

    def setup_fs(self):
        logger.info("Checking environment, please wait...")
        if not self._check_conda():
            raise Exception('The environment is not configured properly.')
        if not self._check_source():
            raise Exception('The FS repo is not configured properly.')


def anonymize(info, psw):
    for key, value in info.items():
        if isinstance(value, dict):
            anonymize(info[key], psw)
        else:
            if key == psw:
                info[key] = "******"
    return info


a = SSHManager('172.17.138.xxx', 'root', 'xxxx', '50012')
