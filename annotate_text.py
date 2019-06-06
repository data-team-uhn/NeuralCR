import argparse
import ncrmodel 
import os
import sys
import json
import csv


#@profile
def annotate_stream(model, threshold, input_iterator, output_writer):
    for i,(key,text) in enumerate(input_iterator):
        sys.stdout.write("\rProgress:: %.2f%%" % (100.0*i//len(input_iterator)))
        sys.stdout.flush()
        ants = model.annotate_text(text,\
                threshold=threshold)
        output_writer.write(key,ants)
    sys.stdout.write("\n")

class DirOutputStream:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    def write(self, key, ants):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_dir+'/'+key,'w') as fp:
            for ant in ants:
                fp.write('\t'.join(map(str,ant))+'\n')

class CSVOutputStream:
    def __init__(self, output_csv_file):
        self.output_csv_file = output_csv_file
        open(self.output_csv_file,'w').close()
    def write(self, key, ants):
        with open(self.output_csv_file,'a') as fw:
            csv_writer = csv.writer(fw, delimiter=';')
            csv_writer.writerow([key]+[str(x) for x in ants])

class DirInputStream:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.filelist = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.filelist)

    def __iter__(self):                           
        self.filelist_iter = iter(self.filelist)
        return self
    
    def __next__(self):                           
        filename = next(self.filelist_iter)
        return filename, open(self.input_dir+'/'+filename).read()

class JsonInputStream:
    def __init__(self, input_json_file):
        with open(input_json_file, 'r') as fp:
            self.notes = json.load(fp)

    def __len__(self):
        return len(self.notes)

    def __iter__(self):                           
        self.notes_iter = iter(self.notes)
        return self

    def __next__(self):                           
        key = self.notes_iter.next()
        return key, self.notes[key]

class CSVInputStream:
    def __init__(self, input_csv_file, max_rows=-1):
        self.input_csv_file = input_csv_file
        if max_rows==-1:
            with open(self.input_csv_file) as fp:
                self.length = sum(1 for row in fp)
        else:
            self.length=max_rows

    def __len__(self):
        return self.length

    def __iter__(self):                           
        self.csvfile = open(self.input_csv_file, 'r') 
        self.reader = csv.reader(self.csvfile, delimiter=',')
        self.csv_iter = iter(self.reader)

        try:
            next(self.csv_iter)
        except StopIteration:
            self.csvfile.close()
            raise StopIteration

        self.ct=0
        return self

    def __next__(self):                           
        self.ct+=1
        if self.ct>self.length:
            self.csvfile.close()
            raise StopIteration

        try:
            row = next(self.csv_iter)
        except StopIteration:
            self.csvfile.close()
            raise StopIteration

        key = row[0]
        text = row[-1]
        return key, text


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--params', help="address to the directroy where the trained model parameters are stored")
    parser.add_argument('--input', help="address to the directory where the input text files are located ")
    parser.add_argument('--output', help="adresss to the directory where the output files will be stored")
    parser.add_argument('--threshold', type=float, help="the score threshold for concept recognition", default=0.8)
    parser.add_argument('--max_rows', type=int, help="", default=-1)
    args = parser.parse_args()

    model = ncrmodel.NCR.loadfromfile(args.params, args.fasttext)
    print ("model loaded")

    if args.output.endswith('.csv'):
        output_stream = CSVOutputStream(args.output)
    else:
        output_stream = DirOutputStream(args.output)

    if args.input.endswith('.csv'):
        input_stream = CSVInputStream(args.input, args.max_rows)
    elif args.input.endswith('.json'):
        input_stream = JsonInputStream(args.output)
    else:
        input_stream = DirInputStream(args.input)
    print ("streams ready")
    annotate_stream(model, args.threshold, input_stream, output_stream)

if __name__ == "__main__":
    main()
