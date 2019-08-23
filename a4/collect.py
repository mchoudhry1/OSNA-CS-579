"""
Collect data.
"""
from TwitterAPI import TwitterAPI
import sys
import time
import pickle

consumer_key = 'L9eODimC2DLk5juHIKKBtRLn4'
consumer_secret = 'Q4EuizjtlHaup2DodEtFzTnd4HNSjS9yJRzTwDBX1dHIEePwy2'
access_token = '1085645241955336194-6uqYZoZQZiO4sPW6SvNZtgpWc8xrHo'
access_token_secret = 'mop2orW0Mqveg4z6CDC35ZoLdBioDAueQHC0ZgPcRClrl'

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    
def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def read_screen_names(filename):
    candidates=[]
    with open(filename, 'r') as file:
        for line in file:
            candidates.append(line.strip())
    file.close()    
    return candidates

def get_friends(twitter, screen_name):
#     names = []
#     for user in screen_name:
    request=robust_request(twitter,"friends/list",{'screen_name':screen_name, 'count' : 100})
#     for i in res:
#         names.append(i["screen_name"])
#     print(len(names))
#     print(names)    
    return request

def hop1_hop2(twitter,screen_name):
    names_1 = []
    answer = {}
    for user in screen_name:
        # print(user)
        request = get_friends(twitter, user)
        names = []
        for friend in request:
            names.append(friend["screen_name"])
        answer[user] = names  
#         print(answer)   
#         print(len(names))
#         print("===========================================")
#     print(answer)
    for i in answer:
        for j in answer[i]:
            names_1.append(j)
    file = open("no_of_users.txt","w") 
    file.write(str(len(names_1)))    
    print("Number of users collected: %d " %(len(names_1)))   
#     for name in names_1:
#         print(name)
#         request_2 = robust_request(twitter,"friends/list",{'screen_name':name,'count':5})
#         names= []
#         for request in request_2:
#             names.append(request["screen_name"])
#         answer[name]= names
    return answer

def write_data_names(friends):
    file = open("names.txt","w")
    for i in friends:
        for j in friends[i]:
            file.write(i+":"+j+"\n")
    file.close()        

def get_tweets(twitter):
    tweets = []
    limit = 2000
    request = robust_request(twitter, 'statuses/filter', {'track':'AvengersEndgame','language':'en','locations':'-88.24617217722074, 41.6163645, -76.2146236146748, 42.746617'})
    for tweet in request:
        tweets.append(tweet)
        if len(tweets) >= limit:
            break
    return tweets

# def get_tweets1(twitter, screen_name, tweets):
#     tweets = list()
#     request = robust_request(twitter, 'search/tweets', {'q': screen_name, 'count': tweets})
#     for tweet in request:
#         tweets.append(tweet['text'])

#     return tweets

def main():
    twitter = get_twitter()
    screen_names = read_screen_names('users.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    write = hop1_hop2(twitter,screen_names)
    write_data_names(write)
    print('Data saved in names.txt. This names.txt file will be used for clustering')
    tweets = get_tweets(twitter)
    length = str(len(tweets))
    print('Number of tweets: %s' %(length))
    pickle.dump(tweets, open('tweets.pkl','wb'))
    print('Tweets saved in tweets.pkl. This file will be used for classfication')
    # tweets1 = get_tweets1(twitter, screen_names[4], 150)
    # len1 = len(tweets1)
    # print('Number of tweets: %d' %(len1))
    # pickle.dump(tweets1, open('tweets1.pkl','wb'))
    # print('Tweets saved in tweets1.pkl. This file will be used for classfication')

if __name__ == "__main__":
    main()