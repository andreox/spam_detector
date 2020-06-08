import sklearn
import nltk
nltk.download('all')
import string
import numpy
import random
import email
import imaplib



dataset_file = open('dataset_spam.txt',encoding="utf8")
dataset = dataset_file.read()
dataset = dataset.lower()

sentences = nltk.sent_tokenize(dataset)

tokens = nltk.word_tokenize(dataset)

pos_tagger = nltk.pos_tag(tokens)


lemmer = nltk.stem.WordNetLemmatizer()
# def nomeFunz(parametri):
def lemTokens(tokens):
  return [lemmer.lemmatize(token, 'v') for token in tokens if token not in set(nltk.corpus.stopwords.words('english'))]

remove_punct_dict = dict( (ord(punct), None) for punct in string.punctuation )
# Definizione della maschera contenente i caratteri di punteggiatura da rimuovere

# Funzione che associa il lemma ed elimina la punteggiatura
def lemNormalize(text):
  return lemTokens( nltk.word_tokenize(text.lower().translate(remove_punct_dict)) )


# TRATTAMENTO DEL TESTO + MODELLO E CALCOLO SUI VETTORI

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def response(email):
  
  sentences.append(email)                 # il chatbot è intelligente ed impara.
                                                  # Quando l'utente inserisce le domande o le frasi, lui le immagazzina

  # Trasforma il testo in matrice mettendo le frasi in riga ed i lemmi in colonna
  cv = CountVectorizer(max_features=100, tokenizer=lemNormalize, analyzer='word')  # Modellazione dell'input
                                        #struttura l'input trasformandolo in un vettore
                                        # Considera come separatore di elementi la funzione lemNormalize
                                          # in questo modo lavora direttamente sui lemmi, e non sulle singole parole
                      # max_features indica il numero di lemmi diversi che possono essere contenuti nelle frasi
                        # quindi se abbiamo 5 frasi, otterremo una matrice di massimo 5x50
  X = cv.fit_transform(sentences)       # Trasforma il testo in una matrice
                                # Questa matrice ha tante righe quante sono le frasi della base di conoscenza, più una
                                # che è l'input dell'utente, ed ha tante colonne quanti sono i lemmi
  # Produco un vettore che ha per ogni componente il coefficiente di similarità tra la riga i-esima della base di conoscenza
  # e l'ultima riga, ovvero la frase relativa alla domanda dell'utente
  vals_cv = cosine_similarity(X[-1],X)
                # Calcola il coefficiente di similarità tra tutta la base di conoscenza (tutte le sentences memorizzate in X)
                # e l'input dell'utente, cioè l'ultima frase immessa dall'utente che è stata inserita con append ed è
                # recuperabile dall'indice -1, che indica l'ultima riga
  frase_piu_simile = vals_cv.argsort()[0][-2]
      # Ordina le frasi per indice crescente di similarità con l'input dell'utente
      # -1 è la frase stessa
      # -2 è la frase più simile alla domanda dell'utente
    
    # Questo ci fa ottenere un vettore, ad esempio di lunghezza 4 per un testo contenente 4 sentences (in cui la quarta è la domanda utente),
    # che ha come valori il valore di cosine similarity. L'ultima posizione ha sempre il valore più alto perchè è il valore di cosine similarity
    # tra la domanda utente e la domanda utente stessa. La penultima posizione del vettore (quella accessa con [-2]) è la frase più simile
  
  # Appiattisci l'array di cosine similarity in un vettore di righe
  flat_vals_cv = vals_cv.flatten()

  # Ordina i valori in ordine crescente di Cosine Similarity
  flat_vals_cv.sort()

  # E memorizza il valore più simile secondo la logica appena descritta
  indice_frase_piu_simile = flat_vals_cv[-2]

  # Se il valore più alto è 0, vuol dire che non c'è stata nessuna corrispondenza tra la base di conoscenza e la domanda dell'utente
  # Il chatbot dovrà quindi riconoscere questa situazione e comunicarla all'utente
  if( indice_frase_piu_simile==0 ):
    result = "Email HAM"
    return result
  
  # Se il valore è maggiore di 0, allora proponi all'utente la frase con massima Cosine Similarity
  result =  "Email SPAM"
  return result



EMAIL = 'iapythontest2020@gmail.com'
PASSWORD = 'passwordsicuranumero1'
SERVER = 'imap.gmail.com'


mail = imaplib.IMAP4_SSL(SERVER)
mail.login(EMAIL, PASSWORD)

mail.select('inbox')

status, data = mail.search(None, 'ALL')

mail_ids = []

for block in data:
    
    mail_ids += block.split()


for i in mail_ids:
    
    status, data = mail.fetch(i, '(RFC822)')

    
    for response_part in data:
       
        if isinstance(response_part, tuple):
            
            message = email.message_from_bytes(response_part[1])

            
            mail_from = message['from']
            mail_subject = message['subject']

           
            if message.is_multipart():
                mail_content = ''

               
                for part in message.get_payload():
                    
                    if part.get_content_type() == 'text/plain':
                        mail_content += part.get_payload()
                        
            else:
                
                mail_content = message.get_payload()

           
            print(f'From: {mail_from}')
            print(f'Subject: {mail_subject}')
            print(f'Content: {mail_content}')
            print(response(mail_content))