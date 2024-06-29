#Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import time
import re
from io import BytesIO
import base64
import json

from sklearn.model_selection import GridSearchCV

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
# nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import BorderlineSMOTE

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

from matplotlib.colors import LogNorm


import seaborn as sns


from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

aspek = ['makanan', 'minuman', 'pelayanan', 'tempat', 'harga']

model_final = {}
vectorizer_final = {}


stop_words = set(stopwords.words('indonesian'))
# list_stopwords_tambahan = ['juga', 'saja', 'aja']
# stop_words.extend(list_stopwords_tambahan)
list_stopwords_baru = stop_words
# list_neg = ['tidak', 'tidaklah','lawan','anti', 'belum', 'belom', 'tdk', 'jangan', 'gak', 'enggak', 'bukan', 'sulit', 'tak', 'sblm']


factory = StemmerFactory()
stemmer = factory.create_stemmer()

#Function
def read_data(data_file):
    df_raw = pd.read_csv(data_file, encoding='latin-1')

    return df_raw

def remove_stopwords(ulasan):
    ulasan_bersih = []
    
    for kata in ulasan:
        if kata not in list_stopwords_baru:
            ulasan_bersih.append(kata)
    
    return ulasan_bersih

def handle_negation(ulasan_lower):
    # Tokenisasi kata
    words = word_tokenize(ulasan_lower)

    # Mengidentifikasi kata negasi
    negation_words = set(['tidak', 'bukan', 'tanpa', 'kurang', 'tidaklah','lawan','anti', 'belum', 'belom', 'tdk', 'jangan', 'gak', 'enggak', 'bukan', 'sulit', 'tak', 'sblm'])

    # Mengidentifikasi stop words Bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    # Inisialisasi variabel untuk menyimpan hasil
    result = []

    # Menggunakan variabel untuk melacak apakah kita berada dalam konteks negasi
    is_negation = False

    # Iterasi melalui setiap kata dalam kalimat
    for word in words:
        # Mengubah status negasi jika menemukan kata negasi
        if word.lower() in negation_words:
            is_negation = not is_negation

        # Menambahkan kata ke hasil dengan memperhatikan status negasi
        if is_negation and word.lower() not in stop_words:
            result.append(f"not_{word}")
        else:
            result.append(word)

    # Menggabungkan kembali kata-kata menjadi kalimat
    handled_text = ' '.join(result)
    return handled_text

def stemming(ulasan_token):
    ulasan_stem = []
    
    for kata in ulasan_token:
        ulasan_stem.append(stemmer.stem(kata))
        
    return ulasan_stem

def ubah_slang(ulasan_stem):
    kamus_slangword = eval(open("C:\\Users\ASUS\slangwords.txt").read()) # Membuka dictionary slangword
    pattern = re.compile(r'\b( ' + '|'.join(kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
    ulasan_no_slang = []
    for kata in ulasan_stem:
        filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
        ulasan_no_slang.append(filteredSlang)
        
    return ulasan_no_slang

def data_preprocessing(df):
    print()
    df['punctual'] = df['ulasan'].str.replace('[^a-zA-Z]+',' ')
    df['lowercase'] = df['punctual'].str.lower()
    df['negasi'] =  [handle_negation(i) for i in df['lowercase']]
    df['token'] = [word_tokenize(i) for i in df['negasi']]
    df['remove_stopwords'] = [remove_stopwords(i) for i in df['token']]
    df['stemmed'] = [stemming(i) for i in df['remove_stopwords']]
    df['ubah_slang'] = [ubah_slang(i) for i in df['stemmed']]
    
    return df

def step_preproccess(raw_docs):
    list_stopword = set(stopwords.words('indonesian'))


    #step by step preprocessing
    step_preproccess = []

    token_0 = word_tokenize(str(raw_docs["ulasan"][1]))
    stop_0 = [word for word in token_0 if word not in list_stopword]
    step_preproccess.append(" ".join(stop_0))

    list_stopword.update(['.', ',', '!', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    punct_0 = [word for word in stop_0 if word not in list_stopword]
    step_preproccess.append(" ".join(punct_0))

    word_token_0 = [string.lower() for string in punct_0]
    step_preproccess.append((word_token_0)) 

    return step_preproccess

def latih_model(df, i, threshold_netral=0.2, threshold_positif=0.3):
    
    list_confusion_matrix = []
    list_heatmap = []

    x = pd.Series([" ".join(i) for i in df['ubah_slang']])
    y = df[i]

    
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(x) 
    

    
    with open('vectorizer_pickle','wb') as b:
            pickle.dump(vectorizer,b)
            
    
     # Resampling menggunakan Tomek Links
    tl = TomekLinks()
    x_resampled, y_resampled = tl.fit_resample(x, y)
    
    # Cek bentuk x_resampled dan y_resampled
    print("Bentuk x_resampled setelah Tomek Links:", x_resampled.shape)
    print("Bentuk y_resampled setelah Tomek Links:", y_resampled.shape)
    
    # Resampling menggunakan SMOTE
    smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)    
    x_resampled, y_resampled = smote.fit_resample(x_resampled, y_resampled)

    print("Bentuk x_resampled setelah SMOTE:", x_resampled.shape)
    print("Bentuk y_resampled setelah SMOTE:", y_resampled.shape)
    

    

    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42)


    
    # Initialize the model_eval dictionary
    model_eval = {
        'C': [],
        'Gamma': [],
        'Akurasi': [],
        'Precision': [],
        'Recall': [],
        'F1-score': []
        }
    
    # Define the parameter grid
    param_grid = {
        'C': [0.01, 0.05, 0.25 , 0.5, 0.75, 1, 10, 100],
        # , 0.25 , 0.5, 0.75, 1, 10, 100
        'gamma': [0.01, 0.1, 1, 10 ],
        # 1, 10
    }

    # Define the scorers
    scorers = {
        'Akurasi': make_scorer(accuracy_score),
        'Precision': make_scorer(precision_score, average='macro'),
        'Recall': make_scorer(recall_score, average='macro'),
        'F1-score': make_scorer(f1_score, average='macro')
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo', random_state=None),
        param_grid,
        scoring=scorers,
        refit='F1-score',
        return_train_score=True,
        cv=10
    )

    # Fit the GridSearchCV object to the data
    grid_search.fit(x_train, y_train)

   
    # Get the results of the grid search
    results_gscv = pd.DataFrame(grid_search.cv_results_)


    # Save the results to a CSV file
    results_gscv.to_csv(f'grid_search_results_{i}.csv', index=False)

    # Train a new SVC model with the best parameters found
    best_params = grid_search.best_params_
    svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], class_weight='balanced', decision_function_shape='ovo', random_state=None)
    svm.fit(x_train, y_train)

    # Update model_eval with the results
    for j in range(len(results_gscv)):
        model_eval['C'].append(results_gscv.loc[j, 'param_C'])
        model_eval['Gamma'].append(results_gscv.loc[j, 'param_gamma'])
        model_eval['Akurasi'].append(results_gscv.loc[j, 'mean_test_Akurasi'])
        model_eval['Precision'].append(results_gscv.loc[j, 'mean_test_Precision'])
        model_eval['Recall'].append(results_gscv.loc[j, 'mean_test_Recall'])
        model_eval['F1-score'].append(results_gscv.loc[j, 'mean_test_F1-score'])

    df_akurasi = pd.DataFrame(model_eval)
    df_akurasi_plot = df_akurasi.copy()
    df_akurasi.sort_values(by=['F1-score'], ascending=False, inplace=True)
    df_akurasi.reset_index(drop=True, inplace=True)
    df_akurasi.to_excel(f'df_akurasi_{i}.xlsx',index=False)

    pivot_table = results_gscv.pivot(index='param_C', columns='param_gamma', values='mean_test_F1-score')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    # , xticklabels=gamma_values, yticklabels=c_values
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.title(f'Heatmap for SVM Aspek {i}')
    plt.grid(True)
    plt.tight_layout()

    accfile = BytesIO()
    plt.savefig(accfile, format='png')
    # plt.savefig(accfile, bbox_inches='tight', format='png')
    accfile.seek(0)
    acc_png = "data:image/png;base64,"
    acc_png += base64.b64encode(accfile.getvalue()).decode('utf-8')

    list_heatmap.append(acc_png)
    

    
     # Mengatur skor yang akan dihitung
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    # Melakukan cross-validation dengan skor yang diinginkan
    results = {}
    for score in scoring:
        if score == 'accuracy':
            score_fn = make_scorer(accuracy_score)
        else:
            score_fn = make_scorer(eval(score.replace('_weighted', '') + '_score'), average='macro')

        cv_result = cross_val_score(svm, x_train, y_train, cv=10, scoring=score_fn)
        results[score] = cv_result.mean()

    # Hasil untuk setiap metrik
    for metric, result in results.items():
        print(f'{metric}: {result}')

    # Juga dapat mencetak tipe dari results
    print(type(results))
               
    # Membuat confusion matrix
    y_pred = svm.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Menampilkan confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis", xticklabels=svm.classes_, yticklabels=svm.classes_)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title(f'Confusion Matrix Aspek {i} ')
#     plt.show()
    
    accfile = BytesIO()
    plt.savefig(accfile, format='png')
    # plt.savefig(accfile, bbox_inches='tight', format='png')
    accfile.seek(0)
    acc_png = "data:image/png;base64,"
    acc_png += base64.b64encode(accfile.getvalue()).decode('utf-8')

    list_confusion_matrix.append(acc_png)
#     print(list_confusion_matrix[0])
    
    # Mendapatkan skor dari decision_function
    decision_scores = svm.decision_function(x_test)

    # Menentukan threshold untuk kelas 'netral'
    netral_threshold = threshold_netral  # Sesuaikan dengan nilai threshold yang Anda inginkan

    # Menggunakan threshold untuk mengubah hasil prediksi 'netral'
    netral_class_index = list(svm.classes_).index('Netral')
    positif_threshold = threshold_positif

    # Menggunakan threshold untuk mengubah hasil prediksi 'netral' dan 'positif'
    netral_class_index = list(svm.classes_).index('Netral')
    positif_class_index = list(svm.classes_).index('Positif')
    
    # Menerapkan threshold untuk hasil prediksi
    y_pred_netral = (decision_scores[:, netral_class_index] > netral_threshold).astype(int)
    y_pred_positif = (decision_scores[:, positif_class_index] > positif_threshold).astype(int)
    
    y_pred = np.where(y_pred_netral == 1, y_pred_netral, y_pred_positif)
    
    with open('svm_pickle','wb') as a:
        pickle.dump(svm, a)
     
    
    return svm, x_test, vectorizer, df_akurasi_plot, y_test, y_pred, list_confusion_matrix, list_heatmap


def proses_kalimat_input(kalimat_input, aspek, vectorizer_final, model_final):
    kalimat_input_punctual = kalimat_input.replace('[^a-zA-Z]+',' ')
    kalimat_input_lowercase = kalimat_input_punctual.lower()
    kalimat_input_token = word_tokenize(kalimat_input_lowercase)
    kalimat_input_remove_stopwords = remove_stopwords(kalimat_input_token)
    kalimat_input_stemmed = stemming(kalimat_input_remove_stopwords)
    kalimat_input_ubah_slang = ubah_slang(kalimat_input_stemmed)
    
    kalimat_input_akhir = [" ".join(kalimat_input_ubah_slang)]
    
    
    print("")
    for i in aspek:
        vecs_input = vectorizer_final[i].transform(kalimat_input_akhir)
        prediksi = model_final[i].predict(vecs_input)
        print(f"Memiliki sentimen {prediksi} dalam aspek {i}")
        
def open_chrome():
    #London Victoria & Albert Museum URL
    # driver = webdriver.Chrome(executable_path= r"E:\Folder Kuliah\Skripsi\Apps\webflask\Aplikasi\chromedriver_win32\chromedriver.exe")
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    # options = webdriver.ChromeOptions()
    # options.add_argument('headless')
    #options.add_argument('window-size=1200x600') # optional

    # options = Options()
    # options.page_load_strategy = 'normal'
    # driver = webdriver.Chrome(options=options)

    # Define the path to the new ChromeDriver executable
    chromedriver_path = "E:\Folder Kuliah\Skripsi\Apps\webflask\Aplikasi\chromedriver_win32\chromedriver.exe"

    # Create a ChromeOptions instance
    chrome_options = webdriver.ChromeOptions()

    # Update the path to the ChromeDriver executable in the service
    chrome_service = Service(chromedriver_path)

    # Initialize the Chrome WebDriver with the updated service
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    url = 'https://www.google.co.id/maps/'
    driver.get(url)
    time.sleep(3)
    
    return driver
    
def search_resto(nama_resto, driver):
    # find the search bar element
    search_bar = driver.find_element(By.XPATH, '//*[@id="searchboxinput"]')

    # input the place name or address
    search_bar.send_keys(nama_resto)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(5)

def redirect_reviewstab(driver):
    print("STATUS = Redirecting to Reviews tab...")
    driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div/div/button[2]').click()
    time.sleep(5)
    print("STATUS = Opening Sorting Menu...")
    driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[7]/div[2]/button/span').click()
    time.sleep(5)
    print("STATUS = Sorting...")
    driver.find_element(By.XPATH,"(//div[@role='menuitemradio'])[2]").click()
    time.sleep(5)

def scroll_reviewstab(driver):
    SCROLL_PAUSE_TIME = 7
    print("STATUS = Start scrolling...")

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    number = 0
    
    while True:
        print("STATUS = scroll...")
        number = number+1

        # Scroll down to bottom
        ele = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]')
        driver.execute_script('arguments[0].scrollBy(0, 5000);', ele)

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        ele = driver.find_element(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]')
        new_height = driver.execute_script("return arguments[0].scrollHeight", ele)

        if number == 10: #NUM ITERATE
            break

        if new_height == last_height:
            break

        last_height = new_height
    
def iteration_allreviews(driver):
    item = driver.find_elements(By.XPATH,'//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[9]')
    review_list = []

    for i in item:
        button = i.find_elements(By.TAG_NAME, 'button')
        for m in button:
            if m.text == "Lainnya":
                m.click()
        time.sleep(5) 
        review = i.find_elements(By.CLASS_NAME, "wiI7pd")
        
        for ulasan in review:
            review_list.append(ulasan.text)
        
    review_list = [i for i in review_list if len(i.split()) >= 2]
            
    df_test_ulasan = pd.DataFrame({'ulasan': review_list})

    
    return df_test_ulasan

def load_model(aspek):
    list_load_model_svm = []
    list_load_vectorizer = []
    for i in aspek:
        with open(f'svm_{i}_pickle','rb') as r:
            svm_p = pickle.load(r)
            list_load_model_svm.append(svm_p)

        with open(f'vectorizer_{i}_pickle','rb') as r:
            vect_p = pickle.load(r)
            list_load_vectorizer.append(vect_p)

    return list_load_model_svm, list_load_vectorizer

def proses_df_input(df , vectorizer_final, model_final, aspek):
    dict_predict = {}
    
    for i in aspek:
        dict_predict[i] = []
        
    for i in df['ubah_slang']:
        kalimat_input_akhir = [" ".join(i)]
        
        for i in dict_predict.keys():
            vecs_input = vectorizer_final[i].transform(kalimat_input_akhir)
            prediksi = model_final[i].predict(vecs_input)
            prediksi = str(list(prediksi))
            prediksi = prediksi.replace("'",'')
            prediksi = prediksi.replace('[','')
            prediksi = prediksi.replace(']','') 
            dict_predict[i].append(prediksi)
    
    for i in dict_predict.keys():
        df[i] = dict_predict[i]  # menambah kolom setiap aspek ke df, dengan isinya list setiap aspek dalam dict
    
    return df

# def conv_df_to_list(df):

#     list_processed_input = []

#     list_ulasan_input = list(df['ulasan'])
#     list_aspek_makanan_input = list(df['makanan'])
#     list_aspek_minuman_input = list(df['minuman'])
#     list_aspek_pelayanan_input = list(df['pelayanan'])
#     list_aspek_tempat_input = list(df['tempat'])
#     list_aspek_harga_input = list(df['harga'])

#     list_processed_input.append(list_ulasan_input)
#     list_processed_input.append(list_aspek_makanan_input)
#     list_processed_input.append(list_aspek_minuman_input)
#     list_processed_input.append(list_aspek_pelayanan_input)
#     list_processed_input.append(list_aspek_tempat_input)
#     list_processed_input.append(list_aspek_harga_input)

#     return list_processed_input

def summary(df, aspek):
    
    dict_sentimen_final = {}
    for i in aspek:
        dict_sentimen_final[f'sentimen_{i}_final'] = df[i].mode()[0]
        
#     sentimen_makanan_final = df['makanan'].mode()[0]
#     sentimen_minuman_final = df['minuman'].mode()[0]
#     sentimen_pelayanan_final = df['pelayanan'].mode()[0]
#     sentimen_tempat_final = df['tempat'].mode()[0]
#     sentimen_harga_final = df['harga'].mode()[0]

    return dict_sentimen_final

def tes_prediksi():
    nama_resto = 'Bakso Aci Juara'
    driver = open_chrome()
    search_resto(nama_resto, driver)
    redirect_reviewstab(driver)
    scroll_reviewstab(driver)
    df_test_ulasan = iteration_allreviews(driver)
    df_test_ulasan

    df_testing = data_preprocessing(df_test_ulasan)
    df_testing = proses_df_input(df_testing)
    dict_sentimen = summary(df_testing)
    print(dict_sentimen)

# def plot_akurasi(list_df_akurasi):
#     list_plot = []

#     for i in range(min(5, len(list_df_akurasi))):
#     # Plot the learning curve
#         x_label = list(list_df_akurasi[i]['C'])
#         # akurasi_label = list_df_akurasi[i]['Akurasi']
#         # precision_label = list_df_akurasi[i]['Precision']
#         # recall_label = list_df_akurasi[i]['Recall']
#         f1_score_label = list_df_akurasi[i]['F1-score']

#         plt.figure(figsize=(10, 8))
#         # plt.plot(x_label, akurasi_label, '-o', label='Akurasi')
#         # plt.plot(x_label, precision_label, '-o', label='Precision')
#         # plt.plot(x_label, recall_label, '-o',  label='Recall')
#         plt.plot(x_label, f1_score_label, '-o', label='F1-Score')

# # akurasi_label,precision_label,recall_label,
#         for y_label in [f1_score_label]:
#             for x,y in zip(x_label,y_label):
#                 label = "{:.2f}".format(y)
#                 plt.annotate(label, # this is the text
#                          (x,y), # this is the point to label
#                          textcoords="offset points", # how to position the text
#                          xytext=(5,5), # distance from text to points (x,y)
#                          ha='center') # horizontal alignment can be left, right or center

#         plt.xlabel('Jumlah C')
#         plt.ylabel('Nilai')
#         #     plt.ylim(ymin=0, ymax=1)
#         plt.title(f'Learning Curve for SVM Aspek {aspek[i]}')
#         plt.grid(True)
#         plt.legend()
#         accfile = BytesIO()
#         plt.savefig(accfile, format='png')
#         # plt.savefig(accfile, bbox_inches='tight', format='png')
#         accfile.seek(0)
#         acc_png = "data:image/png;base64,"
#         acc_png += base64.b64encode(accfile.getvalue()).decode('utf-8')

#         list_plot.append(acc_png)

#     return  list_plot


# def plot_heatmap(list_df_akurasi):
#     list_heatmap = []

#     for i in range((len(list_df_akurasi))):
#         df_akurasi = list_df_akurasi[i]
#         c_values = list(df_akurasi['C'])
#         gamma_values = list(df_akurasi['Gamma'])
#         f1_scores = df_akurasi.pivot(index='C', columns='Gamma', values='F1-score')

#         plt.figure(figsize=(10, 8))
#         sns.heatmap(f1_scores, annot=True, fmt=".2f", cmap="viridis", xticklabels=gamma_values, yticklabels=c_values)
#         plt.xlabel('Gamma')
#         plt.ylabel('C')
#         plt.title(f'Heatmap for SVM Aspek {aspek[i]}')
#         plt.grid(True)
#         plt.tight_layout()

#         accfile = BytesIO()
#         plt.savefig(accfile, format='png')
#         # plt.savefig(accfile, bbox_inches='tight', format='png')
#         accfile.seek(0)
#         acc_png = "data:image/png;base64,"
#         acc_png += base64.b64encode(accfile.getvalue()).decode('utf-8')

#         list_heatmap.append(acc_png)

#     return list_heatmap



def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

def plot_input_rumahmakan(df_test_ulasan):
    
    list_dict = []
    list_plot_input = []
    
    total_review = df_test_ulasan.shape[0]
    df_testing = df_test_ulasan
    print(df_testing.head())
    print(total_review)
    
    list_makanan = df_testing['makanan'].value_counts()
    list_minuman = df_testing['minuman'].value_counts()    
    list_pelayanan = df_testing['pelayanan'].value_counts()
    list_tempat = df_testing['tempat'].value_counts()
    list_harga = df_testing['harga'].value_counts()
    
    print(list_makanan)
    print(list_minuman)
    print(list_pelayanan)
    print(list_tempat)
    print(list_harga)

    dict_makanan = dict(list_makanan)
    dict_minuman = dict(list_minuman)
    dict_pelayanan = dict(list_pelayanan)
    dict_tempat = dict(list_tempat)
    dict_harga = dict(list_harga)
    
    list_dict.append(dict_makanan)
    list_dict.append(dict_minuman)
    list_dict.append(dict_pelayanan)
    list_dict.append(dict_tempat)
    list_dict.append(dict_harga)
    print(list_dict)

    idx = 0
    colours = {'Aspek tidak ada': '#8d7e7c',
           'Positif': '#15b95c',
           'Negatif': '#fb586d',
           'Netral': '#fbab04'}
    
    for i in list_dict:
        names= i.keys()
        values= pd.Series(list(i.values()))
        print(names)
        print(values)
       
        # Label distance: gives the space between labels and the center of the pie
        plt.figure()
        _, _, autotexts = plt.pie(values, labels=names,  autopct=make_autopct(values),wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white'}, colors = [colours[key] for key in names])
        # plt.legend()
        for ins in autotexts:
            ins.set_color('white')
        
        plt.title(f'Ulasan aspek {aspek[idx]} menurut Pengunjung', bbox={'facecolor':'0.8', 'pad':5})
        # plt.show()

        accfile = BytesIO()
        plt.savefig(accfile, format='png')
        accfile.seek(0)
        acc_png = "data:image/png;base64,"
        acc_png += base64.b64encode(accfile.getvalue()).decode('utf-8')
        list_plot_input.append(acc_png)
        idx+=1


    return  list_plot_input, total_review
        

#System

#df = pd.read_csv('dataset_final.csv')

stop_words = set(stopwords.words('indonesian'))
list_stopwords_baru = stop_words
# list_neg = ['tidak', 'tidaklah','lawan','anti', 'belum', 'belom', 'tdk', 'jangan', 'gak', 'enggak', 'bukan', 'sulit', 'tak', 'sblm']
# list_stopwords_baru = list(set(stop_words) - set(list_neg))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

#df = data_preprocessing(df)

# aspek = ['makanan', 'minuman', 'pelayanan', 'tempat', 'harga']
# model_final = {}
# vectorizer_final = {}

# for i in aspek:
    # model, x_test, vectorizer = latih_model(df, i)
    # model_final[i] = model
    # vectorizer_final[i] = vectorizer
