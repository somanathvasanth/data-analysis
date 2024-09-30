import numpy as np
def  UpdateMean(OldMean, NewDataValue, n, A):
    return ((OldMean*n)+NewDataValue)/(n+1)
def UpdateMedian(Oldmedian, NewDataValue, n, A):
    if(n==1):
        return (NewDataValue+A[0])/2
        
    if(n>=2):
        if(n%2==0):
            if(A[n//2]<=NewDataValue):
                return A[n//2]
            if(A[n//2-1]>=NewDataValue):
                return A[n//2-1]
            else :
                return NewDataValue
        if(n%2!=0):
            if(A[n//2-1]>NewDataValue):
                return (A[n//2-1]+A[n//2])/2
            if(A[n//2+1]<NewDataValue):
                return (A[n//2]+A[n//2+1])/2
            else :
                return (NewDataValue+A[n//2])/2

def UpdateStd(OldMean, OldStd, NewMean, NewDataValue, n, A):
    return np.sqrt(((NewDataValue*NewDataValue)+(OldMean*OldMean*n)+(OldStd*OldStd*(n-1))-(NewMean*NewMean*(n+1)))/n)
