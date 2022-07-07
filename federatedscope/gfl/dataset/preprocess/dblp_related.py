from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import re
from bson.json_util import loads

KEYWORDS = [
    'AAAI', 'Association for the Advancement of Artificial Intelligence',
    'CIKM', 'Conference on Information and Knowledge Management', 'CVPR',
    'Conference on Computer Vision and Pattern Recognition', 'ECIR',
    'European Conference on Information Retrieval', 'ECML',
    'European Conference on Machine Learning', 'EDBT',
    'International Conference on Extending Database Technology', 'ICDE',
    'International Conference on Data Engineering', 'ICDM',
    'International Conference on Data Mining', 'ICML',
    'International Conference on Machine Learning', 'IJCAI',
    'International Joint Conference on Artificial Intelligence', 'PAKDD',
    'Pacific-Asia Conference on Knowledge Discovery and Data Mining', 'PKDD',
    'Principles and Practice of Knowledge Discovery in Databases', 'KDD',
    'Knowledge Discovery and Data Mining', 'PODS',
    'Principles of Database Systems'
    'SIGIR', 'Special Interest Group on Information Retrieval', 'SIGMOD',
    'Special Interest Group on Management of Data', 'VLDB',
    'Very Large Data Bases', 'WWW', 'World Wide Web Conference', 'WSDM',
    'Web Search and Data Mining', 'SDM',
    'SIAM International Conference on Data Mining'
]
CONF2ORG = {
    'AAAI': 'AAAI',
    'CIKM': 'ACM',
    'CVPR': 'IEEE',
    'ECIR': 'Springer',
    'ECML': 'Springer',
    'EDBT': 'Springer',
    'ICDE': 'IEEE',
    'ICDM': 'IEEE',
    'ICML': 'PMLR',
    'IJCAI': 'the IJCAI, Inc.',
    'KDD': 'ACM',
    'PAKDD': 'Springer',
    'PKDD': 'Springer',
    'PODS': 'ACM',
    'SDM': 'SIAM',
    'SIGIR': 'ACM',
    'SIGMOD': 'ACM',
    'VLDB': 'VLDB',
    'WWW': 'ACM',
    'WSDM': 'ACM'
}

LABELS = [
    'Database', 'Data mining', 'Artificial intelligence',
    'Information retrieval'
]

parser = argparse.ArgumentParser()
parser.add_argument('--choice', type=int, default=-1)
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
args = parser.parse_args()


def extract_considered():
    keywords = [val.lower() for val in KEYWORDS]
    pat = re.compile(r'|'.join(keywords))

    ent = 0
    cnt = 0
    rsvd = 0
    ops = open(args.output_path, 'w')
    try:
        with open(args.input_path, 'r') as ips:
            ele_contents = []
            is_first = True
            for line in ips:
                if is_first:
                    is_first = False
                    continue

                if line[0] == '{':
                    ent += 1
                elif line[0] == '}':
                    ent -= 1

                ele_contents.append(line.strip())

                if ent == 0 and len(ele_contents):
                    json_text = ''.join(ele_contents)
                    json_text = re.sub(r'NumberInt\s*\(\s*(\S+)\s*\)',
                                       r'{"$numberInt": "\1"}', json_text)
                    # print(json_text[:-1])
                    # ele = json.loads(json_text[:-1])
                    if json_text[-1] == ',':
                        ele = loads(json_text[:-1])
                    else:
                        ele = loads(json_text)
                    # if ('venue' in ele and '_id' in ele['venue']) and
                    # 'fos' in ele and 'references' in ele:
                    if '_id' in ele and 'venue' in ele and 'raw' in ele[
                            'venue'] and ele['venue']['raw'] and 'fos' in \
                            ele and ele[
                                'fos'] and 'references' in ele and 'title' \
                            in ele and ele[
                                    'title']:
                        raw_vanue_name = ele['venue']['raw'].lower()
                        if re.search(pat, raw_vanue_name):
                            ops.write("{}\t{}\t{}\t{}\t{}\n".format(
                                ele['_id'], ele['venue']['raw'].replace(
                                    '\n', '').replace('\t', ' '),
                                ele['title'].replace('\n',
                                                     '').replace('\t', ' '),
                                ','.join(ele['fos']).replace('\n', '').replace(
                                    '\t', ' '), ','.join(ele['references'])))
                            rsvd += 1
                    # print(ele)
                    cnt += 1
                    if cnt % 100000 == 0:
                        print(rsvd, cnt, "======>")
                    ele_contents = []
    except Exception as ex:
        print(ex)
    finally:
        ops.close()


"""
    {'ICDM': 4589, 'KDD': 5476, 'IJCAI': 7586, 'VLDB': 5314, 'PAKDD': 2242,
    'ECIR': 1482, 'ICML': 8322, 'CIKM': 5931, 'WWW': 5553, 'CVPR': 13355,
    'EDBT': 1636, 'AAAI': 9695, 'ECML': 2216, 'SIGMOD': 4206, 'ICDE': 4330,
    'PODS': 1670, 'SDM': 1624, 'SIGIR': 4619, 'WSDM': 746, 'PKDD': 547}
    ======================
    {'IEEE': 22274, 'ACM': 28201, 'the IJCAI, Inc.': 7586, 'VLDB': 5314,
    'Springer': 8123, 'PMLR': 8322, 'AAAI': 9695, 'SIAM': 1624}
"""


def be_canonical():
    keywords = [val.lower() for val in KEYWORDS]
    conf_cnts = dict()
    org_cnts = dict()
    ops = open(args.output_path, 'w')
    with open(args.input_path, 'r') as ips:
        for line in ips:
            num_of_tab = line.count('\t')
            if num_of_tab != 4:
                print(num_of_tab)
                print(line.replace('\t', 'TAB'))
                continue
            cols = line.strip().split('\t')
            conf_raw_name = cols[1].lower()
            org, conf_name = '', ''
            for i, kw in enumerate(keywords):
                if kw in conf_raw_name:
                    conf_name = keywords[i if (i % 2 == 0) else
                                         (i - 1)].upper()
                    org = CONF2ORG[conf_name]
                    break
            if conf_name == '':
                print(cols[1])
                continue
            if conf_name not in conf_cnts:
                conf_cnts[conf_name] = 0
            if org not in org_cnts:
                org_cnts[org] = 0
            conf_cnts[conf_name] += 1
            org_cnts[org] += 1
            ops.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                cols[0], conf_name, org, cols[2], cols[3], cols[4]))
    ops.close()

    print(conf_cnts)
    print("======================")
    print(org_cnts)


def be_fourclass_data():
    labels = [val.lower() for val in LABELS]
    cnt = 0
    vset = dict()
    with open(args.input_path, 'r') as ips:
        for line in ips:
            cols = line.strip().split('\t')
            fos = [val.lower() for val in cols[4].split(',')]
            for val in fos:
                if val in labels:
                    cnt += 1
                    vset[cols[0]] = [0, 0]
                    # assume single label or say the classes are exclusive
                    break
    print(cnt)

    e_cnt = 0
    with open(args.input_path, 'r') as ips:
        for line in ips:
            cols = line.strip().split('\t')
            if cols[0] not in vset:
                continue
            refs = cols[-1].split(',')
            for val in refs:
                if val in vset:
                    e_cnt += 1
                    vset[cols[0]][0] += 1
                    vset[val][1] += 1
    print(e_cnt)

    connected = dict([(val, i) for i, val in enumerate(
        [k for k, v in vset.items() if (v[0] > 0 or v[1] > 0)])])
    print(len(connected))

    ops = open(args.output_path, 'w')
    with open(args.input_path, 'r') as ips:
        for line in ips:
            cols = line.strip().split('\t')
            nid = cols[0]
            if nid not in connected:
                continue
            for val in cols[4].split(','):
                can_val = val.lower()
                if can_val in labels:
                    lb = labels.index(can_val)
                    break
            adjs = ','.join([
                str(connected[val]) for val in cols[-1].split(',')
                if val in connected
            ])
            ops.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                connected[nid], cols[1], cols[2], cols[3], lb, adjs))
    ops.close()


def stats():
    p2c = dict()
    p2o = dict()
    with open(args.input_path, 'r') as ips:
        for line in ips:
            cols = line.strip().split('\t')
            p2c[cols[0]] = cols[1]
            p2o[cols[0]] = cols[2]

    stats = dict()
    with open(args.input_path, 'r') as ips:
        for line in ips:
            cols = line.strip().split('\t')
            conf = cols[1]
            if conf not in stats:
                stats[conf] = [0, 0, 0, [0, 0, 0, 0]]
            stats[conf][0] += 1
            adjs = cols[-1].split(',')
            for v in adjs:
                if p2c[v] == conf:
                    stats[conf][1] += 1
                else:
                    stats[conf][2] += 1
            lb = int(cols[4])
            stats[conf][3][lb] += 1

    for k, v in stats.items():
        print(k, v)

    stats = dict()
    with open(args.input_path, 'r') as ips:
        for line in ips:
            cols = line.strip().split('\t')
            org = cols[2]
            if org not in stats:
                stats[org] = [0, 0, 0, [0, 0, 0, 0]]
            stats[org][0] += 1
            adjs = cols[-1].split(',')
            for v in adjs:
                if p2o[v] == org:
                    stats[org][1] += 1
                else:
                    stats[org][2] += 1
            lb = int(cols[4])
            stats[org][3][lb] += 1

    for k, v in stats.items():
        print(k, v)


def main():
    if args.choice == 0:
        extract_considered()
    elif args.choice == 1:
        be_canonical()
    elif args.choice == 2:
        be_fourclass_data()
    elif args.choice == 3:
        stats()


if __name__ == "__main__":
    main()
