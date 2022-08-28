import pandas as pd #import necessary libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr
from sklearn.metrics import pairwise_distances

userId = 0 #initialize userId variable
itemId = 0 #initialize itemId variable
result = pd.DataFrame(columns=['userId','itemId','predict_score']) # prepare dataframe for saving output

Ratings = pd.read_csv('train.csv',sep=',', names = ["userId","itemId","rating"]) # import train csv

RatingMat = Ratings.pivot_table(index=['itemId'], columns=['userId'], values=['rating'], fill_value=0) # transform to user-item matrix

RatingMat.columns = RatingMat.columns.droplevel()

Original_RatingMat = RatingMat.copy()

RatingMat = RatingMat.apply( lambda x: (x[x!=0] - np.sum(x)/np.count_nonzero(x) ) , axis=1) #subtract row mean

RatingMat = RatingMat.fillna(0.0) # fill null values with 0

Original_RatingMat_numpy = Original_RatingMat.to_numpy(copy = True)

overall_mean_rating = np.sum(Original_RatingMat_numpy) /np.count_nonzero(Original_RatingMat_numpy) # get overall mean rating

item_similarity = cosine_similarity(RatingMat) #get item similarity

test = pd.read_csv('test.csv',sep=',', names = ["userId","itemId"]) #import test set

item_sim_df = pd.DataFrame(item_similarity, index=RatingMat.index, columns=RatingMat.index) # transform item similarity array to dataframe

for i in range(49982): # for loop set to 49982, same as the length of test set

    userId = test.iloc[i]['userId'] #read userId from test dataframe
    itemId = test.iloc[i]['itemId'] #read itemId from test dataframe

    if itemId in item_sim_df.columns: # check if itemId exists in user-item matrix
        curr_item_similarity = pd.DataFrame(item_sim_df.loc[itemId]) #get item similarity for itemId
        Item_Rating = pd.DataFrame(Original_RatingMat.loc[:, userId]) #get user rating for all item
        curr_item_similarity = curr_item_similarity.merge(Item_Rating,right_on='itemId',left_on='itemId') #join the two table based on itemId

        if itemId == userId: # if itemId = userId, then there will be trouble renaming
            itemId2 = str(itemId) + "_x"
            userId2 = str(userId) + "_y"
            curr_item_similarity.rename(columns={itemId2: 'similarity', userId2: 'rating'}, inplace=True)
        else:
            curr_item_similarity.rename(columns = {itemId:'similarity',userId:'rating'}, inplace = True) #renaming columns for data processing

        curr_item_similarity = curr_item_similarity[curr_item_similarity.rating != 0] #remove rows with no rating
        top_k_items = 10 #choose the top 10 items
        curr_item_similarity= curr_item_similarity.sort_values(by='similarity',ascending=False) # sort the dataframe by similarity
        top_k_plus_one_item_similarity = curr_item_similarity.nlargest(n = top_k_items, columns = "similarity") # select the top 10 itemId in terms of similarity

        if top_k_plus_one_item_similarity['similarity'].sum()==0: # check if any similarity exists
            score =0
        else:
            score = (top_k_plus_one_item_similarity['similarity'] * top_k_plus_one_item_similarity['rating']).sum()/(top_k_plus_one_item_similarity['similarity'].sum()) #calculate predict score using CF

    else: score =0

    if score == 0: score = overall_mean_rating # turn all score = 0 to overall mean rating to reduce RMSE
    result = result.append({'userId': userId,'itemId': itemId, 'predict_score': score }, ignore_index=True) # add result to result dataframe

result.to_csv('submit.csv', mode='a', header=False, index=False) # export result dataframe to submit csv
