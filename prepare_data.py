import fasttext_reader
import accuracy


oboFile=open("data/hp.obo")
rd = fasttext_reader.Reader(oboFile)

samplesFile = open("data/labeled_data")
samples = accuracy.prepare_phrase_samples(rd, samplesFile, True)
print rd.name2conceptid
count = 0
print len(samples)
copy_samples = dict(samples)
for x in copy_samples:
    normed = x.lower()
    if normed in rd.name2conceptid:
        del samples[x]
        count += 1
print len(samples)
