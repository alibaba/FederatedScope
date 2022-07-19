import time
import paramiko


class SSHManager(object):
    def __init__(self, ip, user, psw, port):
        self.ssh_port = 22
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
        stdin, stdout, stderr = self.ssh.exec_command(command)
        time.sleep(1)
        output = []
        for line in iter(stdout.readline, ""):
            output.append(line)
        return output

    def setup_fs(self):
        pass


def anonymize(info, psw):
    for key, value in info.items():
        if isinstance(value, dict):
            anonymize(info[key], psw)
        else:
            if key == psw:
                info[key] = "******"
    return info
