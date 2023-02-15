lines=0


W=$(cat /mnt/disk2/ubukata/git_COIL/1/2_for_encoding_corpus/split_chunks.json | wc -l)

# wc -l /mnt/disk2/ubukata/git_COIL/1/2_for_encoding_corpus/split_chunks.json | read num filename

lines=$[lines/100 + 1]
echo ${lines}
echo $W
W=$[W/100 + 1]
echo $W