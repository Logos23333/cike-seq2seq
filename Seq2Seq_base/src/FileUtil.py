#coding: utf-8

import csv

def read_data(filename):
    data_list = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            data_list.append(line)
    return data_list


def load_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        data_list = list(reader)

    msg_list = []
    response_list =[]

    for data in data_list:
        [msg, response, response_, msg_unk, response_unk, response_unk_, symbol_response, symbol_response_, pos, pos_] = data
        msg_unk = msg_unk.lower().strip()
        msg_list.append(msg_unk.split(" "))

        response_unk = response_unk.lower().strip()
        response_list.append(response_unk.split(" "))

    return msg_list, response_list


def read_pairs(filename):
    data_list = read_data(filename)
    pair_list = []

    for data in data_list:
        pairs  = data.strip().split("\t")
        pair_list.append((pairs[0], pairs[1]))
    return pair_list


def write_data(data_list, filename):
    with open(filename, "w",encoding='utf-8') as f:
        for data in data_list:
            f.write(data)


def write_pairs(pair_list, filename):
    data_list = []

    for pair in pair_list:
        if(len(pair[0]) > 0 and len(pair[1]) > 0):
            data_list.append(pair[0].strip() + "\t" + pair[1].strip() + "\n")
    write_data(data_list, filename)