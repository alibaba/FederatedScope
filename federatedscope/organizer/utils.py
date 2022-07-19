import subprocess

ENV_NAME = 'org_test'
DIR = 'org'


class SSHManager(object):
    def __init__(self, ip, user, psw, port, ssh_port=22):
        self.ssh_port = ssh_port
        self.ip = ip
        self.user = user
        self.psw = psw
        self.port = port

    def exec_cmd(self, command):
        """
            Not full login ssh, conduct `source ~/.bashrc` first.
        """
        command = f'source ~/.bashrc; {command}'
        status, output = subprocess.getstatusoutput(f"sshpass -p {self.psw} "
                                                    f"ssh -p {self.ssh_port} "
                                                    f"{self.user}@{self.ip}"
                                                    f" {command}")
        print(status, output)
        return output

    def exec_python_cmd(self, command):
        command = f'source ~/.bashrc; conda activate {ENV_NAME}; ' \
                  f'cd ~/{DIR}/FederatedScope; python {command}'
        status, output = subprocess.getstatusoutput(f"sshpass -p {self.psw} "
                                                    f"ssh -p {self.ssh_port} "
                                                    f"{self.user}@{self.ip}"
                                                    f" {command}")
        print(status, output)
        return output

    def _check_conda(self):
        """
            Check and install conda env.
        """
        output = self.exec_cmd('which conda')
        if output is None:
            # TODO: Install conda here
            pass
        else:
            output = self.exec_cmd('conda env list')
            if ENV_NAME not in output:
                # TODO: Install FS env here
                pass
        return True

    def _check_source(self):
        """
            Check and download FS repo.
        """
        pass

    def setup_fs(self):
        print("Checking environment, please wait...")
        self._check_conda()
        self._check_source()


def anonymize(info, psw):
    for key, value in info.items():
        if isinstance(value, dict):
            anonymize(info[key], psw)
        else:
            if key == psw:
                info[key] = "******"
    return info


a = SSHManager('172.17.138.149', 'root', 'xxxxxx', '50012')
a.setup_fs()
