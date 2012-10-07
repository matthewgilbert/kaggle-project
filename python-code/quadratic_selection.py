#quadratic variable selection


from itertools import combinations

execfile("data_exploring.py")


vdata = data.values
ix = []
n,d = vdata.shape

#there are d*(d-1)/2 different combinations.

qdata = np.zeros( (n, d*(d-1)/2 ) )

for col_pos, comb in enumerate( combinations( range(d), 2) ):
    i = comb[0]
    j = comb[1]
    qdata[:, col_pos] = vdata[:,i]*vdata[:,j]
    ix.append( (i,j) )
    

    
lr = sklm.Lasso( alpha = .001, normalize = True)
lr.fit( qdata, response)

print "Percent of variables selected: %.2f"%( float(np.nonzero( lr.coef_ ).shape[0])/(d*(d-1)/2) )
print "indexes:"
print np.nonzero( lr.coef_ )
