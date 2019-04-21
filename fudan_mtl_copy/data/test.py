# import data_helpers as dh
# res = dh.load_data('SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
# print(res[2][0])
#
#
# import nltk
import tensorflow as tf
import pandas as pd
# a="sdf ds d e12"
# b = nltk.word_tokenize(a)
# print(b)
# print(b.index('e12'))
#
# token = ['1','j','232','j1','23','2546'];
# max_sentence_length = 20
# e1 = 3
# p1 = ''
# for i in range(len(token)):
#     p1 += str((max_sentence_length - 1) + i - e1) + " "
# print(p1)


a = ["1 . 2 , 5","3 2 3 2","2 3 2 4"];
p = tf.contrib.learn.preprocessing.VocabularyProcessor(10)
a = p.fit_transform(a)
print(list(a))
b=["3","l","i","b"]
b = p.fit_transform(b)
print(list(b))
