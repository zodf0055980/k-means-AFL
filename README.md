# k-means-AFL
Use k-means clustering to group seed in seed pool to improve seed selection.

## How to use
It is same as [AFL](https://github.com/google/AFL). Before start afl-fuzz, open a group server.
```
$ python3 group_seed.py [port]
# Another Terminal
$ ./k-means-fuzz -i testcase_dir -o findings_dir -p [port] -- /path/to/program [...params...]
```
