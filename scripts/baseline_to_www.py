with open("/mnt/disk2/ubukata/COIL/www3_data/baselineEng") as input, open("/mnt/disk2/ubukata/COIL/www3_data/baselineEng_after", "w") as output:
    for line in input:
        qid, a, b, c, d, e = line.strip().split(' ')
        output.write('{} {} {} {} {} {}\n'.format(int(qid), a, b, c, d, e))