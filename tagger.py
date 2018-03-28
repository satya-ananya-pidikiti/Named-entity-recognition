import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    

    L = start_scores.shape[0]
  
    N = emission_scores.shape[0]

    #initial=1
    final_out = [[[[] for z in range(0,2)]for y in range(0,L)] for x in xrange(0,N)]

    for i in range(0,L):
        final_out[0][i][0]=start_scores[i]+emission_scores[0][i]
 
    for i in range(1,N):
        for j in range(0,L):
            #print 'HIiii'
            final_out[i][j][0]=-np.inf
            final_out[i][j][1]=0
            for tag in range(0,L):
                curr=final_out[i-1][tag][0]+trans_scores[tag][j]
                if(curr>final_out[i][j][0]):
                    final_out[i][j][0]=curr
                    final_out[i][j][1]=tag
            final_out[i][j][0]+=emission_scores[i][j]
 
 
    tags_list=[]
    #vals_list=[]
    for i in range(0,L):
        tags_list.append(final_out[N-1][i][0]+end_scores[i])
    val_max=max(tags_list)
    tag_max=tags_list.index(val_max)

    last=N-1
    final_tags=[]
    while(last>=0):
        final_tags.append(tag_max)
        tag_max=final_out[last][tag_max][1]
        last=last-1
    ##print final_tags
    return (val_max,final_tags[::-1])
    
    
        

