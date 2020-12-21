import os
import numpy as np
from shutil import copyfile
import  multiprocessing
from sklearn.metrics.pairwise import cosine_similarity

target_dir = "preprocessed_data/rock-songs"
out_directory = 'data/rock-songs'
os.system('rm -r {}'.format(out_directory))
os.system('mkdir {}'.format(out_directory))

def list_compare_dict(target_dir):
    target_dir = os.path.abspath(target_dir)
    file_names = os.listdir(target_dir)
    song_names = [s.split("_")[1].split(".")[0] for s in file_names]
    reduce_dict = {}
    for name, path in zip(song_names, file_names):
        path = os.path.join(target_dir, path)
        if name in reduce_dict:
            reduce_dict[name].append(path)
        else:
            reduce_dict[name] = [path]

    return reduce_dict


def get_unique_song(same_name_song_list : list):

    if len(same_name_song_list) == 1:
        return same_name_song_list

    feature_dict_list = []
    for song in same_name_song_list:
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
            
    final_result = [same_name_song_list[k] for k in keep]
    return final_result


def store_unique_songs(unique_songs, out_directory):
    for p in unique_songs:
        song_name = p.split('/')[-1]
        out_path = os.path.join(out_directory, song_name)
        copyfile(p, out_path)
    

def work_loop(same_name_song_list):
    unique_song_list = get_unique_song(same_name_song_list)
    store_unique_songs(unique_song_list, out_directory)
    return


if __name__ == "__main__":
    reduce_dict = list_compare_dict(target_dir)
    with multiprocessing.Pool(4) as p:
        p.map(work_loop, reduce_dict.values())

