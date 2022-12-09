class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ContrastiveMonitor(object):
    def __init__(self,
                 stat=1,
                 enc_hidden=None,
                 synth_tokens=None,
                 dec_hidden=None,
                 dec_out=None,
                 all_group_ids=None,
                 topk_group_ids=None):
        self.stat = stat
        self.enc_hidden = enc_hidden
        self.synth_tokens = synth_tokens
        self.dec_hidden = dec_hidden
        self.dec_out = dec_out
        self.all_group_ids = all_group_ids
        self.topk_group_ids = topk_group_ids

    def update_stat(self, status):
        self.stat = status

    def update_all_group_ids(self, group_ids):
        self.all_group_ids = group_ids

    def update_topk_group_ids(self, group_ids):
        self.topk_group_ids = group_ids

    def update_enc_hidden(self, enc_hidden, k=None):
        if k is None:
            self.enc_hidden = enc_hidden
        else:
            if self.enc_hidden is None:
                self.enc_hidden = {}
            self.enc_hidden[k] = enc_hidden

    def update_synth_tokens(self, synth_tokens, k=None):
        if k is None:
            self.synth_tokens = synth_tokens
        else:
            if self.synth_tokens is None:
                self.synth_tokens = {}
            self.synth_tokens[k] = synth_tokens

    def update_dec_hidden(self, dec_hidden, k=None):
        if k is None:
            self.dec_hidden = dec_hidden
        else:
            if self.dec_hidden is None:
                self.dec_hidden = {}
            self.dec_hidden[k] = dec_hidden

    def update_dec_out(self, dec_out, k=None):
        if k is None:
            self.dec_out = dec_out
        else:
            if self.dec_out is None:
                self.dec_out = {}
            self.dec_out[k] = dec_out

    def reset(self):
        self.stat = 1
        self.dec_hidden = None
        self.dec_out = None
        self.group_ids = None
