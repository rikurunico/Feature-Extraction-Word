from urllib import response
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_diabetes

dataset = open("code/orange.txt", "r")
finaldata = dataset.readlines()

vectorizer = TfidfVectorizer(stop_words='english')
response = vectorizer.fit_transform(finaldata)
print(response)

response2 = vectorizer.get_feature_names_out()
print(response2)
response2 = pd.DataFrame(response2)
response2.to_csv('code/tokenized.csv', index=False)

response3 = response.todense()
response3 = pd.DataFrame(response3)
response3.to_csv('code/hasilResponse.csv', index=False)
