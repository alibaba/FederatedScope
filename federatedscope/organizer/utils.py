import time
import paramiko

ENV_NAME = 'org_test'
DIR = 'fs_client'


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

    def disconnect(self):
        self.trans.close()

    def exec_cmd(self, command):
        if self.ssh is None:
            self._connect()
            print(f'Connected to {self.ip}.')
        stdin, stdout, stderr = self.ssh.exec_command(command)
        time.sleep(1)
        print(stdout.read(), stderr.read())
        # output = []
        # for line in iter(stdout.readline, ""):
        #     output.append(line)
        return True

    def setup_fs(self):
        print("Installing FederatedScope, please wait...")
        self.exec_cmd(f'cd ~; mkdir -p {DIR}; '
                      f'cd {DIR}; '
                      'git clone -b organizer '
                      'https://github.com/rayrayraykk/FederatedScope.git || '
                      'true; '
                      'cd FederatedScope; '
                      'bash federatedscope/organizer/scripts/install.sh'
                      f' {ENV_NAME}')


def anonymize(info, psw):
    for key, value in info.items():
        if isinstance(value, dict):
            anonymize(info[key], psw)
        else:
            if key == psw:
                info[key] = "******"
    return info


a = SSHManager('172.17.138.149', 'root', 'Dailalgo2022', '50012')
a.setup_fs()
