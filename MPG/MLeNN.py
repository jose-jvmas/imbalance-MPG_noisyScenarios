import numpy as np
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.neighbors import NearestNeighbors

""" MultiLabel edited Nearest Neighbor """
""" MLeNN: A First Approach to Heuristic Multilabel Undersampling (Charte et al., 2014) """
class MLeNN():

    @staticmethod
    def getFileName(param_dict):
        return 'MLeNN_NN-{}_HT-{}'.format(param_dict['NN'], param_dict['HT'])


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['HT'] = 0.5
        param_dict['NN'] = 5
        return param_dict


    """ IRLbl and MeanIR """
    def _imbalanceMetrics(self):

        temp = [np.count_nonzero(self.y[:,it]) for it in range(self.y.shape[1])]
        temp = [1 if (u == 0) else u for u in temp]
        
        self.IRLbl = np.array([max(temp)/temp[u] for u in range(len(temp))])
        self.MeanIR = np.average(self.IRLbl)

        return


    """ Adjusted Hamming distance computation """
    def adjustedHammingDistance(self, y_true, y_pred):

        active_labels = np.count_nonzero(y_true) + np.count_nonzero(y_pred)
        err = hamming_loss(y_true = y_true, y_pred = y_pred) * y_true.shape[0]

        adjustedError = err/active_labels

        return adjustedError

    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.HT = param_dict['HT']
        self.NN = param_dict['NN']
        return


    """ Method for performing the reduction """
    def reduceSet(self, X, y, param_dict):
        self.X = X
        self.y = y

        # Processing parameters:
        self.processParameters(param_dict)

        # Creating search structure:
        searchNN = NearestNeighbors(n_neighbors = self.NN + 1)
        searchNN.fit(self.X)

        # Computing imbalance metrics (IRLbl and MeanIR):
        self._imbalanceMetrics()

        # Initializing list of candidates to remove:
        removeElements = list()

        # Iterating through the samples:
        for it_sample in range(self.X.shape[0]):
            # Checking whether to process the sample based on imbalance ratio (IRLbl):         
            boolProcessSample = True
            if True in list(self.IRLbl[np.where(self.y[it_sample] == 1)[0].tolist()] > self.MeanIR):
                boolProcessSample = False 
            # print("Sample #{} - Process? {}".format(it_sample, boolProcessSample))

            # Process the sample (if so):
            if boolProcessSample == True:
                numDifferences = 0

                for it_NN in searchNN.kneighbors(np.expand_dims(self.X[it_sample], 0))[1][0][1:]:
                    if self.adjustedHammingDistance(y_true = self.y[it_sample], y_pred = self.y[it_NN]) > self.HT:
                        numDifferences += 1


                if numDifferences >= self.NN/2:
                    removeElements.append(it_sample)


        # Remove instances:
        self.X_out = np.delete(self.X, removeElements, axis = 0)
        self.y_out = np.delete(self.y, removeElements, axis = 0)

        return self.X_out, self.y_out




if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = MLeNN.getParameterDictionary()
    X_red, y_red = MLeNN().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)




    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red = cls_red.predict(X_test)

    print("Results:")
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")