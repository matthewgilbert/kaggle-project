#quadratic variable selection



#
# TODO: See if the coefficents of LR change much across the groups.

#
import numpy as np
from itertools import permutations

execfile("data_exploring.py")


vdata = data.values[:, :58]
ix = []
n,d = vdata.shape
print vdata.shape
#there are d*(d-1)/2 different combinations.

qdata = np.zeros( (n, d**2 ) )
k = 0
for comb in permutations( range(d), 2):
    i = comb[0]
    j = comb[1] 
    v = vdata[:,i]*1/vdata[:,j]
    if np.any( np.isnan(v) ) or np.any( np.isinf(v) ):    	
	print "invalid"

    else:
	
	qdata[:, k] =v	
	k+=1
    	ix.append( (i,j) )
	print k

lr = sklm.Lasso( alpha = .001, normalize = True)
lr.fit(qdata, response)
print "fit done"
sparse_ix = [ ix[x] for x in np.nonzero(lr.coef_) ]
for comb in sparse_ix:
	numerator = data.columns[ comb[0] ]
	denominator = data.columns[ comb[1] ]
	print "data['%s'/'%s'] = data['%s']/data['%s']"%( numerator[:10], denominator[:10], numerator, denominator)

