import pandas as pd 
import os

data = pd.read_csv("pp2/votes.csv")
data_boring = data.loc[data['category'] == 'boring']
data_boring["left_image"] = 0 
data_boring["right_image"] = 0

print(data_boring.shape)

for count in range(0, 10):
    count_25000 = count * 12500
    sub_dataframe = data_boring[count_25000:count_25000 + 12500]

    directory = "divided_with_bill/" + str(count)
    os.makedirs(directory)

    sub_dataframe.to_csv(directory + "/vote.csv")



final_dataframe = data_boring[12500*10:]

directory = "divided_with_bill/" + str(10)
os.makedirs(directory)

final_dataframe.to_csv(directory + "/vote.csv")
    
