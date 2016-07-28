import requests
import json
import cPickle as pickle

total=0
hit1 = 0
hit5 = 0
hit10 = 0
ans = {}
for line in open('querries'):
	tokens = line.strip().split("#")
	rep = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+tokens[0].replace(" ","+")).json()

	ans[tokens[0]] = [s[u'id'] for s in rep['rows']]


	top10 = [s[u'id'] for s in rep['rows'][:10]]
	top5 = [s[u'id'] for s in rep['rows'][:5]]
	top1 = [s[u'id'] for s in rep['rows'][:1]]
	if tokens[1] in top10:
		hit10 += 1

	if tokens[1] in top5:
		hit5 += 1

	if tokens[1] in top1:
		hit1 += 1
	
	total += 1

print ans
pickle.dump(ans, open('phenotips_ans.p','wb'))

print total
print float(hit1)/total
print float(hit5)/total
print float(hit10)/total
