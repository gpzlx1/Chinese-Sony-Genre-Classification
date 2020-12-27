# 数据清理部分代码

请注意，该repo中的相关数据已经完成清理，无需再次进行数据清理。

* `TextHandling.py`

    * 去除歌词的发行信息

    * 去除歌词的版权声明

    * 去除不是非常重要的标点符号

    * 繁体歌词化为简体歌词

    * 去除外文歌，保留中文歌

    ```shell
    usage: TextHandling.py [-h] --src-dir SRC_DIR --dst-dir DST_DIR
    ```

* `global_lyric_dedup.py`

    * 歌词去重

    Note : `global_lyric_dedup.py` 消耗比较多的内存。

    ```shell
    usage: global_lyric_dedup.py [-h] --src-dir SRC_DIR --dst-dir DST_DIR
    ```
