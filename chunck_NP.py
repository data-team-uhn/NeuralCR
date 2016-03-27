import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()


def normalise(word):
	"""Normalises words to lowercase and stems and lemmatizes it."""
	word = word.lower()
#    word = lemmatizer.lemmatize(word)
#    word = stemmer.stem_word(word)
	return str(word)

def acceptable_word(word):
	"""Checks conditions for acceptable word: length, stopword."""
	accepted = bool(2 <= len(word) <= 40)
#        and word.lower() not in stopwords)
	return accepted

def tokenize(sentence):
	sentence_re = r'''(?x)      # set flag to allow verbose regexps
		  ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
		| \w+(-\w+)*            # words with optional internal hyphens
		| \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
		| \.\.\.                # ellipsis
		| [][.,;"'?():-_`]      # these are separate tokens
	'''
	return nltk.word_tokenize(sentence) #, sentence_re)
	return nltk.regexp_tokenize(sentence, sentence_re)



def leaves(tree):
	"""Finds NP (nounphrase) leaf nodes of a chunk tree."""
	for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
		yield subtree.leaves()

def get_terms(tree):
	terms=[[ normalise(w) for w,t in leaf if acceptable_word(w) ] for leaf in leaves(tree)]
	return [term for term in terms if len(term)>0]

def chunk_NP(sentence):
	grammar = r"""
		INP:
			{<DT>?(<JJ><CC>?)*<NN.*>+(<IN><DT>?<JJ>*<NN.*>+)*}  # Nouns and Adjectives, terminated with Nouns
		NP:
			{<INP>(<IN><INP>)*}  # Nouns and Adjectives, terminated with Nouns
	"""
	chunker = nltk.RegexpParser(grammar)

	toks = tokenize(sentence)
	postoks = nltk.tag.pos_tag(toks)

	tree = chunker.parse(postoks)
	tree = chunker.parse(tree)

	terms = get_terms(tree)
	return terms

def extract_NP_fromText(text):
	sentences=nltk.sent_tokenize(text)
	return [chunk_NP(sentence) for sentence in sentences]


