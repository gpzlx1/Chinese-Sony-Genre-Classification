import os
import numpy as np
from shutil import copyfile
import queue
import re
from sklearn.metrics.pairwise import cosine_similarity

import argparse

parser = argparse.ArgumentParser(description='Lyrics data clean')
parser.add_argument('--src-dir', default=str, required=True, help='the source directory')
parser.add_argument('--dst-dir', default=str, required=True, help='where to store results (directory)')
args = parser.parse_args()

target_dir_path = args.src_dir
out_dir_parent = args.dst_dir
os.system('mkdir {}'.format(out_dir_parent))


def list_compare_dict(target_dir):
    target_dir = os.path.abspath(target_dir)
    file_names = os.listdir(target_dir)
    file_paths = [os.path.join(target_dir, name) for name in file_names]
    return file_paths


def get_song_feature(song_path):
    out = None
    with open(song_path, 'r') as f:
        out = f.read()
    feature_dict = {}
    for char in out:
        if char >= '\u4e00' and char <= '\u9fa5':
            if char in feature_dict:
                feature_dict[char] = feature_dict[char] + 1
            else:
                feature_dict[char] = 1 

    return feature_dict


def get_unique_song(song_path_list : list):
    
    if len(song_path_list) == 1:
        return song_path_list

    feature_dict_list = []
    for song in song_path_list:
        out = None
        feature_dict = {}
        with open(song, 'r') as f:
            out = f.read()
        for char in out:
            if char >= '\u4e00' and char <= '\u9fa5':
                if char in feature_dict:
                    feature_dict[char] = feature_dict[char] + 1
                else:
                    feature_dict[char] = 1
        feature_dict_list.append(feature_dict)

    total_word = set()
    for s in feature_dict_list:
        for key, _ in s.items():
            total_word.add(key)

    feature_dim = len(total_word)
    print("total word : {}".format(feature_dim))        

    total_word_dict = {}
    for index, word in enumerate(total_word):
        total_word_dict[word] = index

    features = np.zeros((len(feature_dict_list), feature_dim))
    for index, s in enumerate(feature_dict_list):
        for key, value in s.items():
            features[index][total_word_dict[key]] = value

    similarities = cosine_similarity(features)

    keep = set()
    drop = set()
    for i, similarity in enumerate(similarities):
        if i in keep or i in drop:
            continue
        same = []
        for j, s in enumerate(similarity):
            if s >= 0.8:
                same.append(j)
        if len(same) > 1:
            keep.add(same[0])
            for d in same[1:]:
                drop.add(d)
        else:
            keep.add(i)
            
    final_result = [song_path_list[k] for k in keep]
    return final_result

def store_unique_songs(unique_songs, out_dir):
    for p in unique_songs:
        song_name = p.split('/')[-1]
        out_path = os.path.join(out_dir, song_name)
        copyfile(p, out_path)


def main():    
    reduce_paths = list_compare_dict(target_dir_path)
    total = len(reduce_paths)
    print("find song num: {}".format(total))
    unique_song_paths = get_unique_song(reduce_paths)
    print("left", len(unique_song_paths))
    store_unique_songs(unique_song_paths, out_dir_parent)

        
if __name__ == "__main__":
    main()