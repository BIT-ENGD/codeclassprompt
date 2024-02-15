from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer




corpus=['I am a girl adsfa','there is an apple sadfdaf sfd']

vectorizer=CountVectorizer(min_df=1, max_df=1.0)
transformer=TfidfTransformer()

tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
tf_ans=tfidf.toarray()
print(tf_ans)