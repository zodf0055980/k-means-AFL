import sys
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from collections import Counter
from sklearn import cluster, metrics
import socket
import signal

kmeans_group = 30
seed_group = []
for i in range(kmeans_group):
    seed_group.append([])


def signal_handler(sig, frame):
    for i in range(kmeans_group):
        seed_group[i] = sorted(seed_group[i], key=lambda k: k['fuzzcount'])
        print(f"group {i} : {seed_group[i]}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
HOST = '127.0.0.1'
PORT = 7789

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
conn, addr = sock.accept()
init_condition = conn.recv(250).decode("utf-8").split()
dirpath = init_condition[0]
argv = init_condition[1:]
print(dirpath)
print(argv)

ok = conn.recv(8)
print(ok)
seed_list = [os.path.basename(x)
             for x in glob.glob('./'+dirpath+'/queue/id*')]
seed_list.sort()
# obtain raw bitmaps
raw_bitmap = {}
tmp_cnt = []
out = ''

count = 0
for argv_search in argv:
    if argv_search.find('.cur_input') >= 0:
        break
    count = count + 1

for filename in seed_list:
    argv[count] = './'+dirpath+'/queue/' + filename
    print(argv)
    tmp_list = []
    out = subprocess.check_output(
        ['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', 'none', '-t', '500'] + argv)
    for line in out.splitlines():
        edge = line.split(b':')[0]
        tmp_cnt.append(edge)
        tmp_list.append(edge)
    raw_bitmap[filename] = tmp_list
counter = Counter(tmp_cnt).most_common()

# save bitmaps to individual numpy label
label = [int(f[0]) for f in counter]
bitmap = np.zeros((len(seed_list), len(label)))
for idx, i in enumerate(seed_list):
    tmp = raw_bitmap[i]
    for j in tmp:
        if int(j) in label:
            bitmap[idx][label.index((int(j)))] = 1

fit_bitmap, indices = np.unique(bitmap, return_index=True, axis=1)
new_label = [label[f] for f in indices]

kmeans_fit = cluster.KMeans(n_clusters=kmeans_group).fit(fit_bitmap)

cluster_labels = kmeans_fit.labels_
print("====== cluster_labels ======")
print(cluster_labels)
print(len(cluster_labels))

seed_group[cluster_labels[0]].append({"id": 0, "fuzzcount": 1})
for i in range(1, len(cluster_labels)-1):
    seed_group[cluster_labels[i]].append({"id": i, "fuzzcount": 0})
for i in range(kmeans_group):
    seed_group[i] = sorted(seed_group[i], key=lambda k: k['fuzzcount'])
    print(f"group {i} : {seed_group[i]}")

print(f"run rarget = {seed_group[0][0]['id']}")
conn.sendall(str(seed_group[0][0]['id']).encode(encoding="utf-8"))
seed_group[0][0]['fuzzcount'] = seed_group[0][0]['fuzzcount'] + 1
run_group = 0
seed_count = len(cluster_labels)

skip = 3
while(1):
    require = conn.recv(5)
    print(require)
    if(require == b'next'):
        skip = 3
        seed_list = [os.path.basename(x)
                     for x in glob.glob('./'+dirpath+'/queue/id*')]
        # have new path
        if(seed_count < len(seed_list)):
            # sort first
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: k['fuzzcount'])
            # send next seed
            conn.sendall(str(seed_group[run_group]
                             [0]['id']).encode(encoding="utf-8"))
            # fuzz count ++
            seed_group[run_group][0]['fuzzcount'] = seed_group[run_group][0]['fuzzcount'] + 1
            print(f"run target = {seed_group[run_group][0]['id']}")

            # predict
            print(f"find new path {seed_count} to {len(seed_list)}")
            seed_list.sort()
            predic = []
            for i in range(seed_count, len(seed_list)):
                argv[count] = './'+dirpath+'/queue/' + seed_list[i]
                # print(f"predict {seed_list[i]}")
                out = subprocess.check_output(
                    ['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', 'none', '-t', '500'] + argv)
                tmp_list = []
                for line in out.splitlines():
                    edge = line.split(b':')[0]
                    tmp_list.append(edge)
                predict_bitmap = [int(i) for i in tmp_list]
                predicrt_label = []
                for i in new_label:
                    if i in predict_bitmap:
                        predicrt_label.append(1)
                    else:
                        predicrt_label.append(0)
                predic.append(predicrt_label)
            predict_list = kmeans_fit.predict(predic)

            for i in range(len(predict_list)):
                seed_group[predict_list[i]].append(
                    {"id": seed_count + i, "fuzzcount": 0})
                print(f"add {seed_count + i} in group {predict_list[i]}")
            seed_count = len(seed_list)
        # no new path
        else:
            # run next group
            run_group = (run_group + 1) % kmeans_group
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: k['fuzzcount'])
            conn.sendall(str(seed_group[run_group][0]
                             ['id']).encode(encoding="utf-8"))
            # fuzz count ++
            seed_group[run_group][0]['fuzzcount'] = seed_group[run_group][0]['fuzzcount'] + 1
            print(
                f"run next group {run_group} rarget = {seed_group[run_group][0]['id']}")
    elif(require == b'skip'):
        if(skip == 0):
            skip = 3
            run_group = (run_group + 1) % kmeans_group
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: k['fuzzcount'])
            conn.sendall(str(seed_group[run_group][0]
                             ['id']).encode(encoding="utf-8"))
            print(
                f"run next group {run_group} target = {seed_group[run_group][0]['id']}")
            seed_group[run_group][0]['fuzzcount'] = seed_group[run_group][0]['fuzzcount'] + 1
        else:
            skip = skip - 1
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: k['fuzzcount'])
            conn.sendall(str(seed_group[run_group][0]
                             ['id']).encode(encoding="utf-8"))
            print(f"run target = {seed_group[run_group][0]['id']}")
            seed_group[run_group][0]['fuzzcount'] = seed_group[run_group][0]['fuzzcount'] + 1
