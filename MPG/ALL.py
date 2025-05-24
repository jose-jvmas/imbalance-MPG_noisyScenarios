from MPG.MLeNN import MLeNN
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier


""" Exhaustive search case """
class ALL():

    @staticmethod
    def getFileName(param_dict):
        return 'ALL_ENN-{}'.format(param_dict['ENN'])

    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['ENN'] = False # True, False
        return param_dict


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.ENN = param_dict['ENN']
        return


    """ Method for performing MLeNN prior to MPG process """
    def _performMLeNN(self, X, y):
        param_dict = MLeNN().getParameterDictionary()
        X_out, y_out = MLeNN().reduceSet(X, y, param_dict)

        return X_out, y_out
    

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

        return self.X_init, self.y_init



if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    X_red, y_red = ALL().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), 1)

    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red = cls_red.predict(X_test)

    print("Results:")
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")