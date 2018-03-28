#!/bin/python
import nltk
#from nltk.corpus import wordnet as wn
#from nltk.tokenize import TreebankWordTokenizer
#from nltk.corpus import stopwords
#nltk.download('popular')
lexicon_dict={}
#tag_is=[]
#stop_words=set()
def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    #lexicon_dict['stop_words'] = set(open('stop_words').read().split())
    lexicon_dict['people_name']=set(open('data\\lexicon\\firstname.5k').read().title().split())
    lexicon_dict['people_name'].update(set(open('data\\lexicon\\lastname.5000').read().title().split()))
    lexicon_dict['people_name'].update(set(open('data\\lexicon\\people.family_name').read().title().split()))
    lexicon_dict['people_person']=set(open('data\\lexicon\\people.person').read().title().split())
    lexicon_dict['people_name'].update(set(open('data\\lexicon\\people.person.lastnames').read().title().split()))
    
    lexicon_dict['product']=set(open('data\\lexicon\\product').read().title().split())
    lexicon_dict['business_products']=set(open('data\\lexicon\\business.consumer_product').read().title().split())

    lexicon_dict['sports_team']=set(open('data\\lexicon\\sports.sports_team').read().title().split())

    lexicon_dict['tvprog']=set(open('data\\lexicon\\tv.tv_program').read().title().split())
    
    lexicon_dict['museum'] =  set(open('data\\lexicon\\architecture.museum').read().title().split())
    lexicon_dict['auto_make']=set(open('data\\lexicon\\automotive.make').read().title().split())
    lexicon_dict['auto_model']=set(open('data\\lexicon\\automotive.model').read().title().split())
    lexicon_dict['award']=set(open('data\\lexicon\\award.award').read().title().split())
    lexicon_dict['fest_ser']=set(open('data\\lexicon\\base.events.festival_series').read().title().split())
    lexicon_dict['reg_name']=set(open('data\\lexicon\\bigdict').read().title().split())
    lexicon_dict['newspaper']=set(open('data\\lexicon\\book.newspaper').read().title().split())
    lexicon_dict['tv_channels']=set(open('data\\lexicon\\broadcast.tv_channel').read().title().split())
    lexicon_dict['business_brand']=set(open('data\\lexicon\\business.brand').read().title().split())
    lexicon_dict['business_company']=set(open('data\\lexicon\\business.brand').read().title().split())
    lexicon_dict['business_brand']=set(open('data\\lexicon\\business.consumer_company').read().title().split())

    lexicon_dict['business_sponsor']=set(open('data\\lexicon\\business.sponsor').read().title().split())
    lexicon_dict['top10']=set(open('data\\lexicon\\cap.10').read().title().split())
    lexicon_dict['top100']=set(open('data\\lexicon\\cap.100').read().title().split())
    lexicon_dict['cap500']=set(open('data\\lexicon\\cap.500').read().title().split())
    lexicon_dict['cap1000']=set(open('data\\lexicon\\cap.1000').read().title().split())
    lexicon_dict['video_game']=set(open('data\\lexicon\\cvg.computer_videogame').read().title().split())
    lexicon_dict['cvg_developer']=set(open('data\\lexicon\\cvg.cvg_developer').read().title().split())
    lexicon_dict['cvg_platform']=set(open('data\\lexicon\\cvg.cvg_platform').read().title().split())
    #leaving out dictionaries.conf,english.stop,lower.100,lower.500,lower.1000,lower.5000,lower.10000
    lexicon_dict['dictionaries_conf']=set(open('data\\lexicon\\dictionaries.conf').read().title().split())
    lexicon_dict['english_stop']=set(open('data\\lexicon\\english.stop').read().title().split())
    lexicon_dict['lower_10000']=set(open('data\\lexicon\\lower.10000').read().title().split())
    #lexicon_dict['cvg_platform']=set(open('data\\lexicon\\cvg.cvg_platform').read().title().split())
    
    lexicon_dict['university']=set(open('data\\lexicon\\education.university').read().title().split())
    lexicon_dict['gov_agency']=set(open('data\\lexicon\\government.government_agency').read().title().split())


    lexicon_dict['location']=set(open('data\\lexicon\\location').read().title().split())
    lexicon_dict['location'].update(set(open('data\\lexicon\\location.country').read().title().split()))
    lexicon_dict['sports_league']=set(open('data\\lexicon\\sports.sports_league').read().title().split())


    lexicon_dict['time_holiday']=set(open('data\\lexicon\\time.holiday').read().title().split())
    lexicon_dict['time_rec_event']=set(open('data\\lexicon\\time.recurring_event').read().title().split())
    lexicon_dict['roads']=set(open('data\\lexicon\\transportation.road').read().title().split())
    lexicon_dict['tvnet']=set(open('data\\lexicon\\tv.tv_network').read().title().split())

    lexicon_dict['ven_company']=set(open('data\\lexicon\\venture_capital.venture_funded_company').read().title().split())
    lexicon_dict['venues']=set(open('data\\lexicon\\venues').read().title().split())
#sents = [[ "I", "love", "food" ]]
#preprocess_corpus(sents)    
#for key in lexicon_dict:
    #print len(lexicon_dict[key])
def token2features(sent, i, add_neighs = True):
    #print lexicon_dict['people_name1']
    ftrs = []
    # bias
        
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())

    #check if first letter is capitalized
    if word[0].isupper():
        ftrs.append("FIRST_ISCAPS")

    #part of speech
    tag_is= nltk.tag.pos_tag([word])
    #print tag_is
    ftrs.append("POS_TAG="+tag_is[0][1])

    #lexicons
    for key in lexicon_dict:
        if word.title() in lexicon_dict[key]:
            ftrs.append(key.upper())
            #check=False
        #if check=False:
            #break
        
            
    #tag_is= nltk.tag.pos_tag(sent)
    #print tag_is[i][1]
    #ftrs.append("POS_TAG="+tag_is[i][1])

    #checks if first char is @
    if word[0]=='@':
        ftrs.append('FIRST_@')

    #check if first char is num
    if word[0].isdigit():
        ftrs.append('FIRST_NUM')

    #Checks if first is punct:
    if word[0] in [',','.',';','?',':','!','-','\'','"']:
        ftrs.append('FIRST_PUNCT')
    
        
    #check for inner capital
    for l in word[1:]:
        if l.isupper():
            ftrs.append('MIDDLE_CAP')
            break
    #checking last character
    #ftrs.append('LAST_CHAR'+word[len(word)-1])

    #checking first character
    #ftrs.append('FIRST_CHAR'+word[0])

    #word normalization
    curr=''
    for l in word:
        if l.isupper():
            curr+='L'
        elif l.islower():
            curr+='l'
        elif l.isdigit():
            curr+='1'
        else:
            curr+=l
    ftrs.append('NORM='+curr)

    #if it starts with http
    if len(word)>5:
        if word[:6]=='http:':
            ftrs.append('starts with http')
    
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)
    #print lexicon_dict['people_person'][:10]
    # return it!
    
        
    return ftrs

if __name__ == "__main__":
    #Clube
    sents = [
    [ "Bela", "Vista", "Futebol" ,"Clube"]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)

