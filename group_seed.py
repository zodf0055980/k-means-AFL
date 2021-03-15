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

kmeans_group = 0
regroup_count = 300
seed_group = []


def signal_handler(sig, frame):
    for i in range(kmeans_group):
        seed_group[i] = sorted(
            seed_group[i], key=lambda k: (k['fuzzcount'], k['skip']))
        print(f"group {i} : {seed_group[i]}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
HOST = '127.0.0.1'
PORT = int(sys.argv[1])
print(f"[*] port = {PORT}")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
conn, addr = sock.accept()
init_condition = conn.recv(250).decode("utf-8").split()
dirpath = init_condition[0]
argv = init_condition[1:]
print(f"[*] dirpath = {dirpath}")
print(f"[*] argv = {argv}")

init_seed_count = int(conn.recv(8))
seed_list = [os.path.basename(x) for x in glob.glob(dirpath+'/queue/id*')]
seed_list.sort()

kmeans_group = int(len(seed_list) ** 0.5)
print(f"[*] kmeans_group = {kmeans_group}")
# initial seed group
for i in range(kmeans_group):
    seed_group.append([])


# find @@ padding
argv_file_padding = 0
for argv_search in argv:
    if argv_search.find('.cur_input') >= 0:
        break
    argv_file_padding = argv_file_padding + 1

# obtain raw bitmaps
raw_bitmap = {}
tmp_cnt = []
out = ''

for filename in seed_list:
    argv[argv_file_padding] = dirpath+'/queue/' + filename
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
print("[*] cluster_labels")
print(cluster_labels)
print("[*] show group")
for i in range(0, init_seed_count):
    seed_group[cluster_labels[i]].append({"id": i, "skip": 0, "fuzzcount": 1})
for i in range(init_seed_count, len(cluster_labels)):
    seed_group[cluster_labels[i]].append({"id": i, "skip": 0, "fuzzcount": 0})
for i in range(kmeans_group):
    seed_group[i] = sorted(seed_group[i], key=lambda k: k['fuzzcount'])
    print(f"group {i} : {', '.join(str(x['id'])for x in seed_group[i]) }")

print(f"[*] run rarget = {seed_group[0][0]['id']}")
conn.sendall(str(seed_group[0][0]['id']).encode(encoding="utf-8"))
run_group = 0
seed_count = len(cluster_labels)
re_group = seed_count + (seed_count // 2)
max_skip = 2
skip = max_skip
while(1):
    require = conn.recv(5)
    if(require == b'next'):
        seed_group[run_group][0]['fuzzcount'] = seed_group[run_group][0]['fuzzcount'] + 1
        # reset skip
        skip = max_skip
        # get current find path
        seed_list = [os.path.basename(x)
                     for x in glob.glob(dirpath+'/queue/id*')]

        if(len(seed_list) > re_group):  # regroup
            print(f"[*] re group")
            # update
            seed_list.sort()
            all_seed = []
            print(f"old group")
            for i in range(kmeans_group):
                print(f"group {i} : {seed_group[i]}")
                for s in seed_group[i]:
                    all_seed.append(s)
            for i in range(seed_count, len(seed_list)):
                all_seed.append({"id": i, "skip": 0, "fuzzcount": 0})
            all_seed = sorted(all_seed, key=lambda k: k['id'])
            print(all_seed)

            seed_count = len(seed_list)
            re_group = seed_count + (seed_count // 2)

            raw_bitmap = {}
            tmp_cnt = []
            out = ''

            for filename in seed_list:
                argv[argv_file_padding] = dirpath+'/queue/' + filename
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

            fit_bitmap, indices = np.unique(
                bitmap, return_index=True, axis=1)
            new_label = [label[f] for f in indices]
            kmeans_group = int(len(seed_list) ** 0.5)
            print(f"[*] kmeans_group = {kmeans_group}")
            kmeans_fit = cluster.KMeans(
                n_clusters=kmeans_group).fit(fit_bitmap)

            cluster_labels = kmeans_fit.labels_
            print("[*] new cluster_labels")
            print(cluster_labels)

            # initial seed group
            seed_group = []
            for i in range(kmeans_group):
                seed_group.append([])

            for i in range(0, len(cluster_labels)):
                seed_group[cluster_labels[i]].append(all_seed[i])
            print("[*] show new group")
            for i in range(kmeans_group):
                seed_group[i] = sorted(
                    seed_group[i], key=lambda k: (k['fuzzcount'], k['skip']), reverse=False)
                print(
                    f"group {i} : {', '.join(str(x['id'])for x in seed_group[i]) }")

            run_group = 0
            print(f"[*] run rarget = {seed_group[0][0]['id']}")
            conn.sendall(str(seed_group[0][0]['id']).encode(encoding="utf-8"))
        elif(seed_count < len(seed_list)):  # have new path
            # sort first
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: (k['fuzzcount'], k['skip']), reverse=False)
            # send next seed
            conn.sendall(str(seed_group[run_group]
                             [0]['id']).encode(encoding="utf-8"))
            # fuzz count ++
            print(f"[*] run target = {seed_group[run_group][0]['id']}")
            # predict
            print(f"[*] find new path {seed_count} to {len(seed_list)-1}")
            seed_list.sort()
            predic = []
            for i in range(seed_count, len(seed_list)):
                argv[argv_file_padding] = dirpath+'/queue/' + seed_list[i]
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
                    {"id": seed_count + i, "skip": 0, "fuzzcount": 0})
                print(f"add {seed_count + i} in group {predict_list[i]}")
            # update seed count
            seed_count = len(seed_list)
        else:  # no new path
            # run next group
            run_group = (run_group + 1) % kmeans_group
            # prevent seed_group is null
            while not seed_group[run_group]:
                run_group = (run_group + 1) % kmeans_group
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: (k['fuzzcount'], k['skip']), reverse=False)
            conn.sendall(str(seed_group[run_group][0]
                             ['id']).encode(encoding="utf-8"))
            # fuzz count ++
            print(f"[*] next group {run_group}")
            print(f"[*] run target = {seed_group[run_group][0]['id']}")
    elif(require == b'skip'):
        seed_group[run_group][0]['skip'] = seed_group[run_group][0]['skip'] + 1
        if(skip == 0):
            # run bext group and reset skip
            skip = max_skip
            run_group = (run_group + 1) % kmeans_group
            while not seed_group[run_group]:
                run_group = (run_group + 1) % kmeans_group
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: (k['fuzzcount'], k['skip']), reverse=False)
            conn.sendall(str(seed_group[run_group][0]
                             ['id']).encode(encoding="utf-8"))
            print(f"[*] next group {run_group}")
            print(f"[*] run target = {seed_group[run_group][0]['id']}")
        else:
            skip = skip - 1
            seed_group[run_group] = sorted(
                seed_group[run_group], key=lambda k: (k['fuzzcount'], k['skip']), reverse=False)
            # this group is fuzz already
            if(seed_group[run_group][0]['skip'] > 0 or seed_group[run_group][0]['fuzzcount'] > 0):
                print(f"[*] group {run_group} is not interesting")
                run_group = (run_group + 1) % kmeans_group
                while not seed_group[run_group]:
                    run_group = (run_group + 1) % kmeans_group
            conn.sendall(str(seed_group[run_group][0]
                             ['id']).encode(encoding="utf-8"))
            print(f"[*] run target = {seed_group[run_group][0]['id']}")
