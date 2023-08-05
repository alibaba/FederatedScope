import os


def check_gpus():
    if not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True


class GPUManager():
    """
    To automatic allocate the gpu, which returns the gpu with the largest
    free memory rate, unless the specified_device has been set up
    When gpus is unavailable, return 'cpu';
    The implementation of GPUManager is referred to
    https://github.com/QuantumLiu/tf_gpu_manager
    """
    def __init__(self, gpu_available=False, specified_device=-1):
        self.gpu_avaiable = gpu_available and check_gpus()
        self.specified_device = specified_device
        if self.gpu_avaiable:
            self.gpus = self._query_gpus()
            for gpu in self.gpus:
                gpu['allocated'] = False
        else:
            self.gpus = None

    def _sort_by_memory(self, gpus, by_size=False):
        if by_size:
            return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(
                gpus,
                key=lambda d: float(d['memory.free']) / d['memory.total'],
                reverse=True)

    def _query_gpus(self):
        args = ['index', 'gpu_name', 'memory.free', 'memory.total']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(
            ','.join(args))
        results = os.popen(cmd).readlines()
        return [self._parse(line, args) for line in results]

    def _parse(self, line, args):
        numberic_args = ['memory.free', 'memory.total']
        to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').
                                      replace('W', ''))
        process = lambda k, v: (int(to_numberic(v))
                                if k in numberic_args else v.strip())
        return {
            k: process(k, v)
            for k, v in zip(args,
                            line.strip().split(','))
        }

    def auto_choice(self):
        """
        To allocate a device
        """
        if self.gpus is None:
            return 'cpu'
        elif self.specified_device >= 0:
            # allow users to specify the device
            return 'cuda:{}'.format(self.specified_device)
        else:
            for old_infos, new_infos in zip(self.gpus, self._query_gpus()):
                old_infos.update(new_infos)
            unallocated_gpus = [
                gpu for gpu in self.gpus if not gpu['allocated']
            ]
            if len(unallocated_gpus) == 0:
                # reset when all gpus have been allocated
                unallocated_gpus = self.gpus
                for gpu in self.gpus:
                    gpu['allocated'] = False

            chosen_gpu = self._sort_by_memory(unallocated_gpus, True)[0]
            chosen_gpu['allocated'] = True
            index = chosen_gpu['index']
            return 'cuda:{:s}'.format(index)


# for testing
if __name__ == '__main__':

    gpu_manager = GPUManager(gpu_available=True, specified_device=0)
    for i in range(20):
        print(gpu_manager.auto_choice())
