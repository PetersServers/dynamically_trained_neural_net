import requests
import json
from PIL import Image
import urllib.request
import datetime
from config import api_key
import os
from nltk.corpus import wordnet
import warnings
import random as rd



# this is the startpoint to download pictures that i will then use to train a neural net to recognize certain images,
# for each neural net i will download pictures that match the expression, and random pictures that don't match the expression
# then i will label the pictures and train a neural net with the pictures and the values


def _fetch_img(search_term, pos_search=True, num_pages=1):
    image_urls = []
    labels = []
    # print("positive search" if pos_search else "negative search")
    for i in range(num_pages):
        url = f"https://serpapi.com/search?q={search_term}&tbm=isch&ijn={i}&api_key={api_key}"
        response = requests.get(url)
        data = json.loads(response.text)
        image_urls += [val.get('thumbnail') for val in data.get('images_results', [])]
        labels += [1 if pos_search else 0 for _ in range(len(data.get('images_results', [])))]
    return image_urls, labels


def _prepr_store_img(directory, image_urls, labels, searchterm):
    print(f"FETCHING {searchterm} DATA")
    # Save the preprocessed images with labels to the directory
    for i, (url, label) in enumerate(zip(image_urls, labels)):
        try:
            # Download the image
            urllib.request.urlretrieve(url, "image.jpg")
            # Open the image using PIL
            img = Image.open("image.jpg")
            #greyscaling is done later
            '''
            # Convert to greyscale
            img = img.convert('L')
            # Resize the image
            img.thumbnail((256, 256))
            # Save the image
            '''
            img.save(f"{directory}/{searchterm}{i}_{label}.jpg")
        except:
            print(f"{url} number {i} causing problem")

def _check_validity(searchterm, negative_word):
    # could use a word categorization to categorize
    # that mountain is outdoors, and return beach as the
    # negative searchword f.e. increase efficiency?

    # the logic still has to be automated to find the
    # best antagony keyword
    keyword_parts = searchterm.split()
    synonyms = []
    for keyword in keyword_parts:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
    if negative_word in synonyms:
        warnings.warn(f"searchterm {searchterm} and negative word "
                      f"{negative_word} are closely related")


def download_prepr(search_term, data_range, negative_word="random pictures"):
    # public function
    # is used to
    if type(search_term) != str:
        exit("searching for wrong data type")
    # Get current date and time
    now = datetime.datetime.now()
    folder_name = f'{search_term.upper()}.ST_{now.year}{now.month}{now.day}{now.hour}{now.minute}'
    directory = os.path.join(os.getcwd(), "images", folder_name)

    _check_validity(search_term, negative_word)

    # Create the directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    image_urls, labels = _fetch_img(search_term, num_pages=data_range)
    _prepr_store_img(directory, image_urls, labels, search_term)
    image_urls, labels = _fetch_img(negative_word, pos_search=False, num_pages=data_range)
    _prepr_store_img(directory, image_urls, labels, negative_word)



    return directory, folder_name

