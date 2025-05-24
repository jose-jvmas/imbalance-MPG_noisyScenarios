import time
import numpy as np
from MPG.MLeNN import MLeNN
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss, f1_score
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier


""" Multilabel Reduction through Homogeneous Clustering algorithm """
class MRHC():

    @staticmethod
    def getFileName(param_dict):
        return 'MRHC_ENN-{}_Imb-{}_Merging-{}'.format(param_dict['ENN'],\
            param_dict['imb'], param_dict['merging'])

    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['ENN'] = False # True, False
        param_dict['imb'] = 'Base' # None, ExcludingSamples
        param_dict['merging'] = 'Base' # Base, PolicyI, PolicyII
        return param_dict


    """ IRLbl and MeanIR """
    def _imbalanceMetrics(self):

        temp = [np.count_nonzero(self.y_init[:,it]) for it in range(self.y_init.shape[1])]
        temp = [1 if (u == 0) else u for u in temp]
        
        self.IRLbl = np.array([max(temp)/temp[u] for u in range(len(temp))])
        self.MeanIR = np.average(self.IRLbl)

        return


    """ Checking cluster homogeneity """
    def checkClusterCommonLabel(self, in_elements):
        # Checking whether there is a common label in ALL elements in the set:
        common_label_vec = [len(np.nonzero(self.y_toReduce[in_elements,it]==1)[0]) == len(in_elements) for it in range(self.y_init.shape[1])]

        return True if True in common_label_vec else False


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.ENN = param_dict['ENN']
        self.imb = param_dict['imb']
        self.merging = param_dict['merging']
        return


    """ Procedure for generating a new prototype """
    def _generatePrototype(self, indexes):

        r = np.median(self.X_toReduce[indexes], axis = 0)

        r_labelset = list()

        for it_label in range(self.y_toReduce.shape[1]):
            n = len(np.where(self.y_toReduce[indexes, it_label] == 1)[0])
            if self.merging == 'Base':
                r_labelset.append(1) if n > len(indexes)//2 else r_labelset.append(0)
            elif self.merging == 'PolicyI':
                r_labelset.append(1) if n > int(np.floor((len(indexes)//2)/self.IRLbl[it_label])) else r_labelset.append(0)
            elif self.merging == 'PolicyII':
                if n > len(indexes)//2:
                    r_labelset.append(1)
                elif n > 0 and self.IRLbl[it_label] > self.MeanIR:
                    r_labelset.append(1)
                else:
                    r_labelset.append(0)

        return (r, r_labelset)


    """ Method for performing MLeNN prior to MPG process """
    def _performMLeNN(self, X, y):
        param_dict = MLeNN().getParameterDictionary()
        X_out, y_out = MLeNN().reduceSet(X, y, param_dict)

        return X_out, y_out


    """ Method for performing the space splitting stage """
    def _spaceSplittingStage(self):

        Q = list()
        Q.append(list(range(self.X_toReduce.shape[0])))
        CS = list()

        while len(Q) > 0:
            C = Q.pop()
            if self.checkClusterCommonLabel(C) or len(C) == 1:
                CS.append(C)
            else:
                M = list()

                # Obtaining set of label-centroids:
                for it_label in range(self.y_toReduce[C].shape[1]):
                    label_indexes = np.where(self.y_toReduce[C, it_label] == 1)[0]
                    if len(label_indexes) > 0:
                        M.append(np.median(self.X_toReduce[np.array(C)[label_indexes],:], axis = 0))
                M = np.array(M) # label X n_features

                resulting_labels = list(range(len(C)))
                if len(C) > M.shape[0]  and M.shape[0] > 1:
                    # Kmeans with M as initial centroids:
                    kmeans = KMeans(n_clusters = M.shape[0], init = M)
                    kmeans.fit(np.array(self.X_toReduce[C] + 0.001, dtype = 'double'))
                    resulting_labels = kmeans.labels_
                pass

                # Create new groups and enqueue them:
                for cluster_index in np.unique(resulting_labels):
                    indexes = list(np.array(C)[np.where(resulting_labels == cluster_index)[0]])
                    Q.append(indexes)
                pass
            pass

        return CS


    """ Method for performing the reduction """
    def reduceSet(self, X, y, param_dict):
        # Processing parameters:
        self.processParameters(param_dict)


        # Initial assignments:
        self.X_init = X
        self.y_init = y

        # Perform MLeNN process:
        if self.ENN:
            self.X_init, self.y_init = self._performMLeNN(self.X_init,\
                self.y_init)

        # Computing imbalance metrics:
        self._imbalanceMetrics()

        # Checking whether to address data imbalance:
        idx_excluded = list()
        idx_included = list()

        if self.imb == 'Base':
            idx_included = list(range(self.X_init.shape[0]))
        elif self.imb == 'ExcludingSamples':
            # Iterating through the samples:
            for it_sample in range(self.X_init.shape[0]):
                # Checking whether to process the sample based on imbalance ratio (IRLbl):
                if True in list(self.IRLbl[np.where(self.y_init[it_sample] == 1)[0].tolist()] > self.MeanIR):
                    idx_excluded.append(it_sample)
                else:
                    idx_included.append(it_sample)


        # Perform space splitting stage:
        self.X_toReduce = self.X_init[idx_included, :]
        self.y_toReduce = self.y_init[idx_included, :]
        CS = self._spaceSplittingStage()

        # Perform prototype merging stage:
        X_out = list()
        y_out = list()
        for single_cluster in CS:
            if len(single_cluster) > 0:
                prot, labels = self._generatePrototype(single_cluster)
                X_out.append(prot)
                y_out.append(labels)
            pass
        pass

        # Adding formerly excluded labels (if there existed):
        X_out = np.append(np.array(X_out), self.X_init[idx_excluded, :], axis = 0)
        y_out = np.append(np.array(y_out), self.y_init[idx_excluded, :], axis = 0)        

        return X_out, y_out




if __name__ == '__main__':
    start_time = time.time()
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = MRHC.getParameterDictionary()
    params_dict['imb'] = 'Base'
    params_dict['ENN'] = False
    X_red_imbFalse_NoENN, y_red_imbFalse_NoENN = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

    params_dict = MRHC.getParameterDictionary()
    params_dict['imb'] = 'ExcludingSamples'
    params_dict['ENN'] = False
    X_red_imbTrue_NoENN, y_red_imbTrue_NoENN = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)


    params_dict = MRHC.getParameterDictionary()
    params_dict['imb'] = 'Base'
    params_dict['ENN'] = True
    X_red_imbFalse_YesENN, y_red_imbFalse_YesENN = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

    params_dict = MRHC.getParameterDictionary()
    params_dict['imb'] = 'ExcludingSamples'
    params_dict['ENN'] = True
    X_red_imbTrue_YesENN, y_red_imbTrue_YesENN = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)



    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red_imbFalse_NoENN = BRkNNaClassifier(k=1).fit(X_red_imbFalse_NoENN, y_red_imbFalse_NoENN)
    cls_red_imbTrue_NoENN = BRkNNaClassifier(k=1).fit(X_red_imbTrue_NoENN, y_red_imbTrue_NoENN)

    cls_red_imbFalse_YesENN = BRkNNaClassifier(k=1).fit(X_red_imbFalse_YesENN, y_red_imbFalse_YesENN)
    cls_red_imbTrue_YesENN = BRkNNaClassifier(k=1).fit(X_red_imbTrue_YesENN, y_red_imbTrue_YesENN)


    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red_imbFalse_NoENN = cls_red_imbFalse_NoENN.predict(X_test)
    y_pred_red_imbTrue_NoENN = cls_red_imbTrue_NoENN.predict(X_test)
    y_pred_red_imbFalse_YesENN = cls_red_imbFalse_YesENN.predict(X_test)
    y_pred_red_imbTrue_YesENN = cls_red_imbTrue_YesENN.predict(X_test)

    print("Results:")
    print("\t - Hamming Loss (init): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss (imb-None;ENN-False): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red_imbFalse_NoENN), 100*X_red_imbFalse_NoENN.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss (imb-ExclSamples;ENN-False): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red_imbTrue_NoENN), 100*X_red_imbTrue_NoENN.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss (imb-None;ENN-True): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red_imbFalse_YesENN), 100*X_red_imbFalse_YesENN.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss (imb-ExclSamples;ENN-True): {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red_imbTrue_YesENN), 100*X_red_imbTrue_YesENN.shape[0]/X_train.shape[0]))
    print("\t" + "---" * 20)
    print("\t - F1 (init): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_ori, average = 'macro'), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - F1 (imb-None;ENN-False): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_red_imbFalse_NoENN, average = 'macro'), 100*X_red_imbFalse_NoENN.shape[0]/X_train.shape[0]))
    print("\t - F1 (imb-ExclSamples;ENN-False): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_red_imbTrue_NoENN, average = 'macro'), 100*X_red_imbTrue_NoENN.shape[0]/X_train.shape[0]))
    print("\t - F1 (imb-None;ENN-True): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_red_imbFalse_YesENN, average = 'macro'), 100*X_red_imbFalse_YesENN.shape[0]/X_train.shape[0]))
    print("\t - F1 (imb-ExclSamples;ENN-True): {:.3f} - Size: {:.1f}%".format(f1_score(y_true = y_test, y_pred = y_pred_red_imbTrue_YesENN, average = 'macro'), 100*X_red_imbTrue_YesENN.shape[0]/X_train.shape[0]))
    print("Done!")
    print("--- %s seconds ---" % (time.time() - start_time))