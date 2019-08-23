"""
Summarize data.
"""
import pickle


def main():
    file = open("summary.txt","w", encoding= "utf-8")
    file.write('\n==============================================')
    file.write("\nCollect.py answers")
    name4 = "users.txt"
    f4 = open(name4, 'r')
    file.write('\n')
    file.write(f4.read())
    file.write('\nThe above screen_names were used and their friends were collected.')
    file.write('\n')
    file.write('Number of users collected: ')
    name = "no_of_users.txt"
    f = open(name,"r")
    file.write(f.read())
    tweets1 = pickle.load(open('tweets.pkl', 'rb'))
    tweets2 = [t for t in tweets1]
    file.write("\nFound %d tweets" %(len(tweets2)))
    file.write('\n==============================================')
    file.write('\nClusters.py answers\n')
    name2 = "cluster_answers.txt"
    f2 = open(name2,"r")
    file.write(f2.read())
    file.write('\n==============================================')
    file.write("\nclassify.py answers")
    file.write('\n')
    name3 = "pos_neg_tweets.txt"
    name5 = "gender_labels.txt"
    f3 = open(name3,"r", encoding = "utf-8")
    f5 = open(name5,"r", encoding = "utf-8")
    file.write(f3.read())
    file.write('\n')
    file.write('gender labels: ')    
    file.write(f5.read())
    file.write('\n')   
    file.write('-1 is unknow, 0 is male, 1 is female ') 
    Answers = pickle.load(open('Answers.pkl', 'rb'))
    file.write('\nAverage Accuracy of Gender Classfication is:%s ' %(str(Answers)))
    file.write('\n==============================================')

if __name__ == "__main__":
    main()