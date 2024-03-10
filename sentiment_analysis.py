import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# function to clean data by converting to lower, remove stoplist items
# and remove any leading, and trailing whitespaces. Add cleaned and 
# original  data to  dictionnary 
def clean_text(whole_review_data):
    cleaned_reviews = {}
    for review in whole_review_data:
        doc = nlp(review)
        cleaned_text = ' '.join([token.text.lower() for token in doc if not 
                                 token.is_stop and token.text.strip()])
        cleaned_reviews[review] = cleaned_text
    return cleaned_reviews

# function to test sentiment in our cleaned data.
def sentiment_analysis(cleaned_data):
    for key, value in cleaned_data.items():
        doc = nlp(value)
        print(f"\nReview: {key}")
        print(f"Sentiment: {doc._.blob.polarity}")
        
# Load Amazon data, remove nan values from reviews column, and filter 
# to a test sample
amazon_data = pd.read_csv("amazon_product_reviews.csv", sep=",")
amazon_data = amazon_data.dropna(subset=['reviews.text'])
reviews_data = amazon_data['reviews.text'].iloc[[0, 24, 43, 437, 2000, 6000, 9000, 11501, 
                                                 13503, 17000, 25066]]

sentiment_analysis(clean_text(reviews_data))


first_review_of_choice = amazon_data['reviews.text'][1500]
second_review_of_choice = amazon_data['reviews.text'][2200]

nlp = spacy.load('en_core_web_md')

# function to test similarity between 2 sentences. 
def similarity(first, second):
    similarity_result = nlp(first).similarity(nlp(second))
    return(similarity_result)


print(f"\nReview One: {first_review_of_choice}")
print(f"Review Two: {second_review_of_choice}")
print(f"Similarity: {similarity(first_review_of_choice,second_review_of_choice)}")
