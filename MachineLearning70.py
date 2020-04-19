import numpy as np
from operator import itemgetter
import csv,os

class KNNClassifier(object):
    def __init__(self):#constructor
        self.training_features=None   #movie kicks kisses {[3,4,5,]}
        self.training_labels=None   #movie gener[Romance,Comedy],label=output of trining,acutual
        self.test_features=None         #kicks=18 kisses =90 [18,90]
        #Build Meaningful result
        self.elegantResult="Most Likely {0},{1} is of type"#elegantresult is the unknowdata
    def loadTrainingFromFile(self,file_path):
        if file_path is not None and os.path.exists(file_path):#the file name is there in os
            tr_features=[]
            self.training_labels=[]
            with open(file_path,'r') as training_data_file:
                reader=csv.DictReader(training_data_file)#read all line other the 1st line
                for row in reader:#divide all the data excpect 1st row
                    if row['moviename']!='?':
                        tr_features.append([float(row['kicks']),float(row['kisses'])])
                        #input feacture of 1st input
                        self.training_labels.append(row['movietype'])
                    else:
                        self.test_features=np.array([float(row['kicks']),float(row['kisses'])])
            if len(tr_features)>0:
                self.training_features=np.array(tr_features)
            print("self.training features   :  ",self.training_features)
            print("self.training labels      :",self.training_labels)
            print("self.test features    :",self.test_features)

            #[1,1,1] #[10,90]    #k=5
    def classifyTestData(self,test_data=None,k=0):#here the test_data is local varible,none is used her for internal
        print("classifytestdata:   test_data  =",test_data)#
        if test_data is not None:
            self.test_features=np.array(test_data,dtype=float)
        print("classify test data:    self.test_features  =",self.test_features)

        #ensure we have training data, training labels and testdata and ....
        if self.test_features is not None and self.training_features is not None and self.training_labels is not None:

            print("classifytestdata says self.testfeatures  :",self.test_features)#[18,90]
            print("self.training_features   :",self.training_features)#[[3,104],...
            print("self.training_labels        :",self.training_labels)#['romance'....]
            featureVectorSize=self.training_features.shape[0]# answer will be 6,number of rows
            print("feature Vector Size    :",featureVectorSize)#array of column(features)
            tileofTestData=np.tile(self.test_features,(featureVectorSize,1))#make 6 row in 1 value ,repeact 6 times
            print("after Tile of test data  :\n",tileofTestData)
            diffMat=self.training_features-tileofTestData
            print("diffMat   :",diffMat)#diffrence matrix
            sqDifMat=diffMat**2
            print("sqDifMat     :",sqDifMat)      #6x2
            sqDistances=sqDifMat.sum(axis=1)       #6x1 pruduce the row
            print("RowwiseSum SqDistances    :",sqDistances)
            distances=sqDistances**0.5
            print("distances(Sq.Root of SqDistances ::  ",distances)
            sortedDistanceIndices=distances.argsort()
            print("sortedDistanceIndices   :::",sortedDistanceIndices)
            print("self.training_labels   :   ",self.training_labels)
            classCount={}
            for i in range(k):      #k=5 ==  0,1,2,3,4
                print("sortedDistanceIndices[i]    :",sortedDistanceIndices[i])
                voteILabel=self.training_labels[sortedDistanceIndices[i]]
                print("voteILabel    :",voteILabel)
                classCount['voteILabel']=classCount.get(voteILabel,0)+1
            #classCount={Action:2 , Romance:3}

            print("classCount   =  ",classCount)
            sortedclassCount=sorted(classCount.items(),key=itemgetter,reverse=True)

            # sortedClassCount = {"Roamnce",3),(" Action" ,2}
            print("sortedclassCount    :",sortedclassCount)
            print("sortedclassCount[0]    :",sortedclassCount[0])
            print("sortedclassCount[0][0]     :",sortedclassCount[0][0])
            return sortedclassCount[0][0]
        else:
            return "Can't Determine result for applying test_data"
def predictMovieType():
    instance=KNNClassifier()
    instance.loadTrainingFromFile("LgR_Movies_kNN_classifier.csv")
    print("********************************************************")
    #my_test_data=[50,50]   #can be supplied to instance classifyTestData()
    #classOfTest_data=instance.classifyTestData(test_data=my_test_data,k=5)
    classOfTest_data=instance.classifyTestData(test_data=None,k=5)
    print("predictMovieType ClassofTestdata",classOfTest_data)

    return instance.elegantResult.format(('movie'),str(instance.test_features),classOfTest_data)
if __name__=="__main__" :
    print(predictMovieType())
