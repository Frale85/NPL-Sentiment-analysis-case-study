import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import wordcloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

#Intro:
#il progetto in esame valuta ed esplora un dataset composto da una raccolta dati,
#in cui sono stati raccolti messaggi di tipo spam e messaggi conformi (cosidetti ham).
#dopo aver eseguito preprocessing, pulizia e organizzazione del dataset
#si passa alla tokenizzazione,statistica dei dati presenti.
#infine si utilizza il metodo di Bayes e il metodo Sgd per addestrare un modello predittivo

#this project evaluates and explores a dataset consisting of a collection
#in which spam and compliant messages (so-called ham) have been collected.
#after preprocessing, cleaning and organising the dataset
#we move on to the tokenization, statistics of the data present.
#finally the Bayes method and the Sgd method are used to train a predictive model



# leggo il dataset denominato df tramite pandas
df = pd.read_csv(r'C:\Users\Francesco L\Desktop\text analytics\text_anl.progetto\spamlist.csv', encoding='latin-1')

print('prime righe del dataframe', df.head())

#Prepocessing e datamining
#i valori importanti per proseguire le analisi sono racchiuse nella prima colonna
#dunque le altre colonne possono essere rimosse e rinominate v1 con "etich" e v2 come "text"
# in seguito raggruppo e vedo le statistiche descrittive    
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"etich", "v2":"text"})
df.describe()

df.groupby("etich").describe()

#Andiamo a vedere ora la distribuzione delle variabili target cercando di plottare il tutto
df.etich.value_counts()
df.etich.value_counts().plot.bar()

#va ad associare delle etichette numeriche agli spam (1) e agli regular (0) per poter classificare il tutto
df['spam'] = df['etich'].map( {'spam': 1, 'ham': 0} ).astype(int)
df.head(10)

#Aggiungo una colonna 'length' che rappresenta la lunghezza dei messaggi.
df['length'] = df['text'].apply(len)
df.head(10)

#Vediamo di graficare con un istogramma la distribuzione esaminata
df.hist(column='length',by='etich',bins=60,figsize=(12,4))
plt.xlim(-40,950)

#divido il dataset in base a spma ed ham
df_ham  = df[df['spam'] == 0].copy()
df_spam = df[df['spam'] == 1].copy()

#tramite la funzione show_wordcloud genero e visualizzo il cloud di parole.
def show_wordcloud(df_spam_or_ham, title):
    text = ' '.join(df_spam_or_ham['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='lightgrey',
                    colormap='viridis', width=800, height=600).generate(text)
    
    plt.figure(figsize=(10,7), frameon=True)
    plt.imshow(fig_wordcloud)  
    plt.axis('off')
    plt.title(title, fontsize=20 )
    plt.show()

#eseguo la funzione precedente per gli ham
show_wordcloud(df_ham, "Messaggi tipo Regolari-Ham")    

#eseguo la funzione precedente per gli spam
show_wordcloud(df_spam, "Messagi tipo Spam")

#inserisco i moduli: string (con i caratteri comuni di punteggiatura da rimuovere)
#stopwords (parole comuni che spesso non contribuiscono a significati dalla libreria NLTK (Natural Language Toolkit)
#Counter(che conta il numero di occorrenze degli elementi in una lista)

import string
string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


#da stringa di testo (mess), rimuove la punteggiatura w suddivide la stringa in una lista di parole
#rimuovendo le stopwords.
# Il testo viene restituito come una lista di parole senza punteggiatura e stopwords.
stopwords.words("english")[100:110]
def remove_punctuation_and_stopwords(mess):
    
    mess_no_punctuation = [ch for ch in mess if ch not in string.punctuation]
    mess_no_punctuation = "".join(mess_no_punctuation).split()
    
    mess_no_punctuation_no_stopwords = \
        [word.lower() for word in mess_no_punctuation if word.lower() not in stopwords.words("english")]
        
    return mess_no_punctuation_no_stopwords

#La funzione viene applicata alla colonna 'text' di df, restituendo una nuova colonna con il testo modellato
df['text'].apply(remove_punctuation_and_stopwords).head()

# creo due liste di parole per i messaggi Ham e Spam, applicando la funzione precedente 
from collections import Counter
df_ham.loc[:, 'text'] = df_ham['text'].apply(remove_punctuation_and_stopwords)
words_df_ham = df_ham['text'].tolist()
df_spam.loc[:, 'text'] = df_spam['text'].apply(remove_punctuation_and_stopwords)
words_df_spam = df_spam['text'].tolist()

#Utilizzo i contatori per esaminare numero di occorrenze di ciascuna parola nei messaggi Ham e Spam.
list_ham_words = []
for sublist in words_df_ham:
    for item in sublist:
        list_ham_words.append(item)

list_spam_words = []
for sublist in words_df_spam:
    for item in sublist:
        list_spam_words.append(item)

c_ham  = Counter(list_ham_words)
c_spam = Counter(list_spam_words)

#creO due DatFrame con 40 parole più comuni in Ham e Spam, insieme al conteggio delle loro occorrenze.
df_hamwords_top40  = pd.DataFrame(c_ham.most_common(40),  columns=['word', 'count'])
df_spamwords_top40 = pd.DataFrame(c_spam.most_common(40), columns=['word', 'count'])


fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', data=df_hamwords_top40, ax=ax, order=df_hamwords_top40['word'])
plt.title("Le 40 parole Ham piu comuni sono")
plt.xticks(rotation='vertical')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', data=df_spamwords_top40, ax=ax, order=df_spamwords_top40['word'])
plt.title("Le 40 parole Spam piu comuni sono")
plt.xticks(rotation='vertical')

plt.show()

#Creo una distribuzione di frequenza per le parole nei messaggi Ham e spam
fdist_ham  = nltk.FreqDist(list_ham_words)
fdist_spam = nltk.FreqDist(list_spam_words)

#creo due DataFrame con le info ricercate
df_hamwords_top40_nltk  = pd.DataFrame(fdist_ham.most_common(40),  columns=['word', 'count'])
df_spamwords_top40_nltk = pd.DataFrame(fdist_spam.most_common(40), columns=['word', 'count'])

#grafico con degli istogrammi le parole per le suddivisioni che ho eseguito in precedenza
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=df_hamwords_top40_nltk, ax=ax)
plt.title("Le 40 parole 40 Ham piu comuni")
plt.xticks(rotation='vertical')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='word', y='count', 
            data=df_spamwords_top40_nltk, ax=ax)
plt.title("Le 40 parole Spam piu comuni")
plt.xticks(rotation='vertical')

#tramite sklearn creo un modello di bag of word
#Specificando una funzione che rimuove punteggiatura e stopwords.
#stampampando lunghezza del vocabolario (num tot di parole uniche nei messaggi)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(df['text'])
print('il num tot di parole uniche nei messaggi :', len(bow_transformer.vocabulary_))

#Eseguo una vettorizzazione di un messaggio di Spam e uno di Ham (sample_ham)
#utilizzando il modello Bag of Words creato.
#Stampo il messaggio originale, la rappresentazione vettoriale sparsa e le parole corrispondenti.
sample_spam = df['text'][10]
bow_sample_spam = bow_transformer.transform([sample_spam])
print('esempio di messaggio originale: ', sample_spam)
print('esempiodi bow spam: ', bow_sample_spam)

rows, cols = bow_sample_spam.nonzero()
for col in cols: 
    print('Nomi features corrispondenti di ogni colonna non nulla: ', bow_transformer.get_feature_names_out()[col])

print('matrice sparsa: ', np.shape(bow_sample_spam))

sample_ham = df['text'][6]
bow_sample_ham = bow_transformer.transform([sample_ham])
print('campione di messaggio estratto dalla sesta riga di df: ',sample_ham)
print(bow_sample_ham)

#itero attraverso le colonne non nulle della matrice sparsa, ottenuta dalla trasformazione del messaggio sample_ham
rows, cols = bow_sample_ham.nonzero()
for col in cols: 
    print('il nome delle feature corrispondente: ', bow_transformer.get_feature_names_out()[col])
    

#Applico la Bow a tutti i messaggi
#inltre restituisco una matrice sparsa dove ogni riga rappresenta un messaggio
#e ogni colonna rappresenta una parola unica con il numero totale di elementi non nulli.
bow_df = bow_transformer.transform(df['text'])
bow_df.shape
bow_df.nnz

#Calcola e stampa la sparsità della matrice spars (la percentuale di valori non nulli rispetto al totale possibile)
bow_df
bow_df.shape[0]
bow_df.shape[1]
bow_df.nnz
print('la percentuale di valori non nulli rispetto al totale possibile: ', bow_df.nnz / (bow_df.shape[0] * bow_df.shape[1]) *100 )

#Crea un trasformatore TF-IDF e lo adatta alla matrice BoW
#in seguito restituisco la rappresentazione TF-IDF di un campione di messaggio Ham e di uno Spam
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(bow_df)

tfidf_sample_ham = tfidf_transformer.transform(bow_sample_ham)
print(tfidf_sample_ham)

tfidf_sample_spam = tfidf_transformer.transform(bow_sample_spam)
print(tfidf_sample_spam)
df_tfidf = tfidf_transformer.transform(bow_df)
df_tfidf
np.shape(df_tfidf)

#eseguo l'addestramento del modello
#per dividere la nuova matrice (X2) e le etichette di spam 
from sklearn.model_selection import train_test_split

df_tfidf_train, df_tfidf_test, etich_train, etich_test = \
    train_test_split(df_tfidf, df["spam"], test_size=0.3, random_state=5)

df_tfidf_train

df_tfidf_test

from scipy.sparse import  hstack
X2 = hstack((df_tfidf ,np.array(df['length'])[:,None])).A

X2_train, X2_test, y2_train, y2_test = \
    train_test_split(X2, df["spam"], test_size=0.3, random_state=5)

#Utilizzo ora una classificazione tramite il metodo Naive Bayes
#in primis trasformo le matrici TF-IDF, da matrici sparse a array
#poi con il modello MNB analizzo la matrice TF-IDF senza scaling.
#Calcolo e stampo l'accuratezza del modello sui dati di test.
#Uso il Min-Max Scaler per standardizzare (scaling) le matrici TF-IDF di addestramento e test.
#Creo un modello MNB e lo addestra utilizzando la matrice TF-IDF con scaling.
#Calcolo e stampa l'accuratezza del modello sui dati di test.
#Utilizzo train_test_split per creare un set di addestramento e test.


df_tfidf_train = df_tfidf_train.A
df_tfidf_test = df_tfidf_test.A 

spam_detect_model = MultinomialNB().fit(df_tfidf_train, etich_train)
pred_test_MNB = spam_detect_model.predict(df_tfidf_test)
acc_MNB = accuracy_score(etich_test, pred_test_MNB)
print('accuracy con Mnb: ', acc_MNB)

scaler = MinMaxScaler()
df_tfidf_train_sc = scaler.fit_transform(df_tfidf_train)
df_tfidf_test_sc  = scaler.transform(df_tfidf_test)

spam_detect_model_minmax = MultinomialNB().fit(df_tfidf_train_sc, etich_train)
pred_test_MNB = spam_detect_model_minmax.predict(df_tfidf_test_sc)
acc_MNB = accuracy_score(etich_test, pred_test_MNB)
print('accuracy min_max con Mnb: ',acc_MNB)

from sklearn.model_selection import train_test_split

mess_train, mess_test, etich_train, etich_test = \
    train_test_split(df["text"], df["spam"], test_size=0.3, random_state=5)

mess_train.head()

# Calcolo della matrice di confusione
conf_matrix = confusion_matrix(etich_test, pred_test_MNB)
print("Matrice di Confusione:\n", conf_matrix)

# Calcolo di precision, recall e F1-score
print("\nReport di Classificazione:\n", classification_report(etich_test, pred_test_MNB))

# Calcolo dell'area sotto la curva ROC
roc_auc = roc_auc_score(etich_test, pred_test_MNB)
print("\nAUC-ROC Score:", roc_auc)


# Crea un modello SGDClassifier con addestramento
sgd_classifier = SGDClassifier()
sgd_classifier.fit(df_tfidf_train, etich_train)

# predizioni del modello
pred_test_SGD = sgd_classifier.predict(df_tfidf_test)

# Calcola e stampa l'accuratezza del modello sui dati di test
acc_SGD = accuracy_score(etich_test, pred_test_SGD)
print('Accuracy con SGD: ', acc_SGD)

