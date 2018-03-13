import numpy as np

class Ontology():

    def dfs(self, c, kids, mark):
	mark.add(c)
	for kid in kids[c]:
            if kid not in mark:
                self.dfs(kid, kids, mark)

    def read_oboFile(self, oboFile, topid=None):
        names={}
        def_text={}
        kids={}
        parents={}
        real_id = {}
        while True:
            line=oboFile.readline()
            if line == "":
                break
            tokens=line.strip().split(" ")
            if tokens[0]=="id:":
                hp_id=tokens[1]
                parents[hp_id] = []
                kids[hp_id] = []
                names[hp_id] = []
                real_id[hp_id] = hp_id

            if tokens[0] == "def:":
                def_text[hp_id] = [line[line.index("\"")+1:line.rindex("\"")]]
            if tokens[0] == "name:":
                names[hp_id] = [' '.join(tokens[1:])]
            if tokens[0] == "synonym:":
                last_index = (i for i,v in enumerate(tokens) if v.endswith("\"")).next()
                names[hp_id].append( ' '.join(tokens[1:last_index+ 1]).strip("\"") )
            if tokens[0] == "alt_id:":
                real_id[tokens[1]] = hp_id
        
        oboFile.seek(0)
        while True:
            line=oboFile.readline()
            if line == "":
                break
            tokens=line.strip().split(" ")
            if tokens[0]=="id:":
                hp_id=tokens[1]

            if tokens[0]=="is_a:":
                kids[tokens[1]].append(hp_id)
                parents[hp_id].append(tokens[1])
        mark=set()
        self.dfs(topid, kids, mark)
        names = {c:names[c] for c in mark}
        def_text = {c:def_text[c] for c in mark if c in def_text}
        parents = {c:parents[c] for c in mark}
        kids = {c:kids[c] for c in mark}
        for c in parents:
            parents[c]=[p for p in parents[c] if p in mark]
        total_names = []
        for c in names:
            for name in names[c]:
                total_names.append(name)
                #print name
#	print len(total_names)
        return names, kids, parents, real_id, def_text


    def _update_ancestry(self, c):
        cid = self.concept2id[c]
        if np.sum(self.ancestry_mask[cid]) > 0:
            return self.ancestry_mask[cid]

        self.ancestry_mask[cid,cid] = 1.0

        for p in self.parents[c]:
            self.ancestry_mask[cid, self.concept2id[p]] = 1.0
            self.ancestry_mask[cid,:]=np.maximum(self.ancestry_mask[cid,:], self._update_ancestry(p))

        return self.ancestry_mask[cid,:]

    def _update_ancestry_sparse(self, c):
        cid = self.concept2id[c]
        if cid in self.ancestrs:
            return self.ancestrs[cid]

        self.ancestrs[cid] = set([cid])

        for p in self.parents[c]:
            self.ancestrs[cid].update(self._update_ancestry_sparse(p))
        return self.ancestrs[cid]



    def __init__(self, obo_address, root_id):
        self.names, self.kids, self.parents, self.real_id, self.def_text = self.read_oboFile(open(obo_address), root_id)
        self.concepts = [c for c in self.names.keys()]
        self.concept2id = dict(zip(self.concepts,range(len(self.concepts))))

        self.name2conceptid = {}
        for c in self.concepts:
            for name in self.names[c]:
                normalized_name = name.strip().lower()
                self.name2conceptid[normalized_name] = self.concept2id[c]

        #self.ancestry_mask = np.zeros((len(self.concepts), len(self.concepts)))
        self.ancestrs = {}
        self.samples = []
        for c in self.concepts:
            #self._update_ancestry(c)
            self._update_ancestry_sparse(c)

        self.sparse_ancestrs = []
        for cid in self.ancestrs:
            #self.sparse_ancestrs += [[ancid, cid] for ancid in self.ancestrs[cid]]
            self.sparse_ancestrs += [[cid, ancid] for ancid in self.ancestrs[cid]]

    def save(self):
        """save class as self.name.txt"""
        file = open(self.name+'.txt','w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open(self.name+'.txt','r')
        dataPickle = file.read()
        file.close()

def main():
    ont = Ontology()
    print len(ont.concepts)

if __name__ == "__main__":
	main()
