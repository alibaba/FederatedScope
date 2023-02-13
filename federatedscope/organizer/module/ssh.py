import os
import paramiko

from federatedscope.organizer.cfg_client import ENV_NAME, ROOT_PATH, FS_VERSION
from federatedscope.organizer.utils import OrganizerLogger


class SSHManager(object):
    def __init__(self, ip, user, psw, ssh_port=22):
        self.logger = OrganizerLogger()
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
        command = f'source ~/.bashrc; conda activate {ENV_NAME}; ' \
                  f'cd ~/{ROOT_PATH}/FederatedScope; {command}'
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
            self.logger.error(
                'Exception: No conda, please install conda first.')
            # TODO: Install conda here
            return False

        # Check env & FS
        output, err = self._exec_cmd(f'conda activate {ENV_NAME}; '
                                     f'python -c "import federatedscope; '
                                     f'print(federatedscope.__version__)"')
        if err:
            self.logger.error(f'Error: {err}')
            # TODO: Install FS env here
            return False
        elif output != FS_VERSION:
            self.logger.info(f'The installed FS version is {output}, however'
                             f' {FS_VERSION} is required.')
        self.logger.info(f'Conda environment found, named {ENV_NAME}.')
        return True

    def _check_source(self):
        """
            Check and download FS repo.
        """
        fs_path = os.path.join(ROOT_PATH, 'FederatedScope', '.git')
        output, _ = self._exec_cmd(f'[ -d {fs_path} ] && echo "Found" || '
                                   f'echo "Not found"')
        if output == 'Not found':
            # TODO: git clone here
            self.logger.error(f'Exception: Repo not find in {fs_path}.')
            return False
        self.logger.info(f'FS repo Found in {ROOT_PATH}.')
        return True

    def _check_task_status(self):
        """
            Check task status.
        """
        pass

    def setup_fs(self):
        self.logger.info("Checking environment, please wait...")
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
