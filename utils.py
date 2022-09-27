import re
import gensim
from pymystem3 import Mystem
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from fse import IndexedList
from fse.models import Average
from gensim.models import KeyedVectors
import pandas as pd


bad_symbols_re = re.compile('[,.«»!"#$%&\'()*+/:;<=>?@[\\]^_`{|}~]')
stopwords = stopwords.words(['russian', 'english'])
mystem = Mystem()

def _text_preprocess(text):
    def _get_word(analysis):
        if analysis.get('analysis'):
            if len(analysis.get('analysis', [])) != 0:
                return analysis.get('analysis', [{}])[0].get('lex', '')
            else:
                return analysis.get('text', '')
        else:
            return analysis.get('text', '')

    text = text.lower()
    text = bad_symbols_re.sub('', text)
    text = ''.join([_get_word(word) for word in mystem.analyze(text)]).strip()
    text = ' '.join([token for token in text.split(' ') if token not in stopwords])

    return text

def _text_preprocess_stem(text, stemmer):

    text = text.lower()
    text = bad_symbols_re.sub('', text)
    text = ' '.join([stemmer.stem(word) for word in text.split(' ') if word not in stopwords])

    return text

def _train_w2v_model(namings_data):
    words = list(map(lambda x: x.split(' '), namings_data.values()))

    model = gensim.models.Word2Vec(
        sentences=words,
        vector_size=300,
        window=5,
        workers=1,
        min_count=5,
        epochs=25,
        seed=42,
        sg=1,
        negative=5,
    )
    model.init_sims(replace=True)

    model = Average(model)
    model.train(IndexedList(words))

    model.sv.vocab = dict(zip(namings_data.keys(), model.sv.vectors))
    
    vector_length = 300
    model_final = KeyedVectors(vector_length)

    key_list = list(model.sv.vocab.keys())
    vector_list = list(model.sv.vocab.values())

    model_final.add_vectors(key_list, vector_list)

    return model_final



def _get_kv_scores_v2(issue_id, model, topn=10):
    result = model.most_similar(model.vectors[model.key_to_index[issue_id]],
                                       topn=topn+1)
    result = dict(zip(list(list(zip(*result))[0]), list(list(zip(*result))[1])))
    if issue_id in result:
        result.pop(issue_id)
    else:
        result.popitem()
    return result

def _get_possible_products_pairs_v2(issue_id, model, topn=10):
    return _get_kv_scores_v2(issue_id, model, topn)

def time_features(df):
    
    # cast date
    df.created = pd.to_datetime(df.created)

    # created month / day
    df["year"] = df.created.dt.year
    df["weekofyear"] = df.created.dt.isocalendar().week
    df["month"] = df.created.dt.month
    df["day"] = df.created.dt.day
    df["dayofweek"] = df.created.dt.dayofweek
    df["hour"] = df.created.dt.hour
    
    return df