# Import google_streetview for the api module
import google_streetview.api
import pandas as pd
import numpy as np
from PIL import Image
import urllib
import time
import cv2
import hashlib
import hmac
import base64
import urlparse

# put your key here ########################################################
streetview_key = "AIzaSyCIbkhjhLwFrhpSINf8CRh-Ws1RokfSgPs" 
secret_key = "5J4_FV-otnknYsdXM8IVPtFBk3c="
sub_count = 2
############################################################################


# cd to directory and get data
directory = "divided_with_bill/" + str(sub_count) + "/"
data = pd.read_csv(directory + "vote.csv")

#streetview secret
def sign_url(input_url=None, secret=None):

  url = urlparse.urlparse(input_url)
  url_to_sign = url.path + "?" + url.query
  decoded_key = base64.urlsafe_b64decode(secret)
  signature = hmac.new(decoded_key, url_to_sign, hashlib.sha1)
  encoded_signature = base64.urlsafe_b64encode(signature.digest())

  original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query
  return original_url + "&signature=" + encoded_signature



count = 0 
for index, row in data.iterrows():

    # left
    left_lat = row["left_lat"]
    left_long = row["left_long"]

    # download left image
    left_location = str(left_lat) + "," + str(left_long)
    left_filename = str(left_lat) + "_" + str(left_long) + ".jpg"
    signed_url_left = sign_url(input_url="https://maps.googleapis.com/maps/api/streetview?size=400x300&location=" + str(left_location) + "&key=" + str(streetview_key), secret=secret_key)
    urllib.urlretrieve(signed_url_left, directory + left_filename)

    # convert to array
    left_image_array = [cv2.imread(directory + "/" + left_filename)]
    data['left_image'] = data['left_image'].astype(object)
    data.loc[index, "left_image"] = left_image_array

    count += 1
    print (count)

    # # right
    right_lat = row["right_lat"]
    right_long = row["right_long"]

    # download right image
    right_location = str(right_lat) + "," + str(right_long)
    right_filename = str(right_lat) + "_" + str(right_long) + ".jpg"
    signed_url_right = sign_url(input_url="https://maps.googleapis.com/maps/api/streetview?size=400x300&location=" + str(right_location) + "&key=" + str(streetview_key), secret=secret_key)
    urllib.urlretrieve(signed_url_right, directory + right_filename)

    # convert to array
    right_image_array = [cv2.imread(directory + "/" + right_filename)]
    data['right_image'] = data['right_image'].astype(object)
    data.loc[index, "right_image"] = right_image_array

    count += 1
    print (count)

data.to_csv(directory + "vote_with_image.csv")


# put your key here ########################################################
streetview_key = "AIzaSyAudpvhpBa7CQCU1-s2gJB5F_OWrbNJvR4"
secret_key = "2Ps8ja5pkR6ZBHTflPsLBzhUV_U="
sub_count = 3
############################################################################

# cd to directory and get data
directory = "divided_with_bill/" + str(sub_count) + "/"
data = pd.read_csv(directory + "vote.csv")

#streetview secret
def sign_url(input_url=None, secret=None):

  url = urlparse.urlparse(input_url)
  url_to_sign = url.path + "?" + url.query
  decoded_key = base64.urlsafe_b64decode(secret)
  signature = hmac.new(decoded_key, url_to_sign, hashlib.sha1)
  encoded_signature = base64.urlsafe_b64encode(signature.digest())

  original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query
  return original_url + "&signature=" + encoded_signature



count = 0 
for index, row in data.iterrows():

    # left
    left_lat = row["left_lat"]
    left_long = row["left_long"]

    # download left image
    left_location = str(left_lat) + "," + str(left_long)
    left_filename = str(left_lat) + "_" + str(left_long) + ".jpg"
    signed_url_left = sign_url(input_url="https://maps.googleapis.com/maps/api/streetview?size=400x300&location=" + str(left_location) + "&key=" + str(streetview_key), secret=secret_key)
    urllib.urlretrieve(signed_url_left, directory + left_filename)

    # convert to array
    left_image_array = [cv2.imread(directory + "/" + left_filename)]
    data['left_image'] = data['left_image'].astype(object)
    data.loc[index, "left_image"] = left_image_array

    count += 1
    print (count)

    # # right
    right_lat = row["right_lat"]
    right_long = row["right_long"]

    # download right image
    right_location = str(right_lat) + "," + str(right_long)
    right_filename = str(right_lat) + "_" + str(right_long) + ".jpg"
    signed_url_right = sign_url(input_url="https://maps.googleapis.com/maps/api/streetview?size=400x300&location=" + str(right_location) + "&key=" + str(streetview_key), secret=secret_key)
    urllib.urlretrieve(signed_url_right, directory + right_filename)

    # convert to array
    right_image_array = [cv2.imread(directory + "/" + right_filename)]
    data['right_image'] = data['right_image'].astype(object)
    data.loc[index, "right_image"] = right_image_array

    count += 1
    print (count)

data.to_csv(directory + "vote_with_image.csv")