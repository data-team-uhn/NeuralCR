from extractor import ExtractModel, ExtConfig
from phrase_model import NCRModel
import phraseConfig
import fasttext
from onto import Ontology

class SentAnt:
    def __init__(self, phrase_model, extractor):
        self.phrase_model = phrase_model
        self.extractor = extractor

    def process_text(self, text, threshold=0.5, is_pheno_threshold=0.5):
        chunks_large = text.replace("\r"," ").replace("\n"," ").replace("\t", " ").replace(",","|").replace(";","|").replace(".","|").split("|")
        candidates = []
        candidates_info = []
        total_chars=0
        for c,chunk in enumerate(chunks_large):
            tokens = chunk.split(" ")
            chunk_chars = 0
            for i,w in enumerate(tokens):
                phrase = ""
                for r in range(7):
                    if i+r >= len(tokens) or len(tokens[i+r])==0:
                        break
                    if r>0:
                        phrase += " " + tokens[i+r]
                    else:
                        phrase = tokens[i+r]
                    #cand_phrase = phrase.strip(',/;-.').strip()
                    cand_phrase = phrase
                    if len(cand_phrase) > 0:
                        candidates.append(cand_phrase)
                        location = total_chars+chunk_chars
                        candidates_info.append((location, location+len(phrase), c))
                chunk_chars += len(w)+1
            total_chars += len(chunk)+1

        is_phenotype_score = self.extractor.predict(candidates)
        new_candidates = [x for i,x in enumerate(candidates) if is_phenotype_score[i]>is_pheno_threshold]
        new_candidates_info = [x for i,x in enumerate(candidates_info) if is_phenotype_score[i]>is_pheno_threshold]

        candidates = new_candidates
        candidates_info = new_candidates_info

        matches = [x[0] for x in self.phrase_model.get_hp_id(candidates, 1)]
        filtered = {}
        #print "---->>>>"
        #print matches
        #print " "
        for i in range(len(candidates)):
            if matches[i][0]!='HP:0000118' and matches[i][0]!="HP:None" and matches[i][1]>threshold:
                if candidates_info[i][2] not in filtered:
                    filtered[candidates_info[i][2]] = []
                filtered[candidates_info[i][2]].append((candidates_info[i][0], candidates_info[i][1], matches[i][0], matches[i][1]))

        #print " "
        final = []
        for c in filtered:
            tmp_final = []
            cands = sorted(filtered[c], key= lambda x:x[0]-x[1])
            for m in cands:
                conflict = False
                for m2 in tmp_final:
                    if m[1]>m2[0] and m[0]<m2[1]:
                        conflict = True
                        break
                if conflict:
                    continue
                best_smaller = m
                for m2 in cands:
                    if m[0]<=m2[0] and m[1]>=m2[1] and m[2]==m2[2] and (m2[1]-m2[0]<best_smaller[1]-best_smaller[0]):
                        best_smaller = m2
                tmp_final.append(best_smaller)
            final+=tmp_final
        return final

def main():
    word_model = fasttext.load_model('data/model_pmc.bin')
    ont = Ontology('data/hp.obo',"HP:0000118")
    ncrmodel = NCRModel(phraseConfig.Config, ont, word_model)
    ncrmodel.load_params('checkpoints/')

    extractmodel = ExtractModel(ExtConfig, word_model)
    extractmodel.load_params('ext_params')

    sentant = SentAnt(ncrmodel, extractmodel)

    text ="renal investigation"
    #text ="spotty bleeding mucosa"
    #text ="spotty bleeding mucosa extending"
    print extractmodel.predict([text])
    print ncrmodel.get_hp_id([text], count=5)
#    print sentant.process_text(full_text)

if __name__ == "__main__":
    main()
