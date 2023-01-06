"""
The implementations are adapted from https://github.com/tylin/coco-caption/
blob/master/pycocoevalcap/meteor/meteor.py
"""

import os
import subprocess
import tempfile

METEOR_JAR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'meteor-1.5.jar')


class Meteor(object):
    """
    The implementation of Meteor refer to 'METEOR: An automatic metric for MT
    evaluation with improved correlation with human judgments'
    [Banerjee S, Lavie A., 2005]
    (https://aclanthology.org/W05-0909.pdf)
    """
    def __init__(self):
        self.meteor_cmd = ' '.join([
            'java', '-Xmx2G', '-jar', METEOR_JAR, '{pred}', '{reference}',
            '-l', 'en', '-norm'
        ])

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        # Clean up a NamedTemporaryFile on your own
        # delete=True means the file will be deleted on close
        pred_tmp = tempfile.NamedTemporaryFile(mode='w', dir='./', delete=True)
        ref_tmp = tempfile.NamedTemporaryFile(mode='w', dir='./', delete=True)
        for i in imgIds:
            assert (len(res[i]) == 1)  # only one prediction per example
            # do stuff with temp
            pred_tmp.write('{}\n'.format(res[i][0]))
            ref_tmp.write('{}\n'.format(gts[i][0]))

        pred_tmp.flush()
        ref_tmp.flush()

        output = subprocess.getoutput(
            self.meteor_cmd.format(pred=pred_tmp.name, reference=ref_tmp.name))
        score = float(output.split('\n')[-1].split(':')[-1].strip())
        pred_tmp.close()  # deletes the file
        ref_tmp.close()  # deletes the file

        return score, None

    def method(self):
        return "METEOR"
