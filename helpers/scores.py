import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay,roc_auc_score,f1_score,precision_score, recall_score,multilabel_confusion_matrix, confusion_matrix,f1_score, accuracy_score,hamming_loss


class Scores:
    def column(matrix, i):
        return [row[i] for row in matrix]
    
    def get_tp_tn_fn_fp(y_true, y_pred):
        cm = []
        for i in range(y_true.shape[1]):
            y1 = Scores.column(y_true, i)
            y2 = Scores.column(y_pred, i)
            tp,tn,fn,fp = 0,0,0,0
            for x in range(len(y1)):
                if ((y1[x] == 1) & (y2[x] == 1)): tp+=1
                if ((y1[x] == 0) & (y2[x] == 1)): fp+=1
                if ((y1[x] == 1) & (y2[x] == 0)): fn+=1
                if ((y1[x] == 0) & (y2[x] == 0)): tn+=1
            temp = [tp,fp,fn,tn]
            cm.append(temp)
        return cm
    
    def compute_tp_tn_fn_fp(y_true, y_pred):
        '''
        True positive - actual = 1, predicted = 1
        False positive - actual = 0, predicted = 1
        False negative - actual = 1, predicted = 0
        True negative - actual = 0, predicted = 0
        '''
        cm = []
        for i in range(y_true.shape[1]):
            y1 = Scores.column(y_true, i)
            y2 = Scores.column(y_pred, i)
            tp,tn,fn,fp = 0,0,0,0
            for x in range(len(y1)):
                if ((y1[x] == 1) & (y2[x] == 1)): tp+=1
                if ((y1[x] == 0) & (y2[x] == 1)): fp+=1
                if ((y1[x] == 1) & (y2[x] == 0)): fn+=1
                if ((y1[x] == 0) & (y2[x] == 0)): tn+=1
            temp = {
                'tp':tp,
                'fp':fp,
                'fn':fn,
                'tn':tn,
            }
            cm.append(temp)
        return cm
    
    def compute_precision(tp, fp, zero_division=0):
        '''Precision = TP  / FP + TP '''
        if (tp + fp == 0 and zero_division == 1):
            return 1
        elif (tp + fp == 0 and zero_division == 0):
            return 0
        else:
            return float(tp)/ float( tp + fp)
    
    def compute_recall(tp, fn, zero_division=0):
        '''Recall = TP /FN + TP'''
        if (tp + fn == 0 and zero_division == 1):
            return 1
        elif (tp + fn == 0 and zero_division == 0):
            return 0
        else:
            return float(tp)/ float( tp + fn)

    def Precision(y_true, y_pred, average='macro', zero_division=0):
        cm = Scores.compute_tp_tn_fn_fp(y_true, y_pred)
        if (average == 'macro'):
            temp = 0
            for index in range(len(cm)):
                tp = cm[index]['tp']
                fp = cm[index]['fp']
                fn = cm[index]['fn']
                tn = cm[index]['tn']
                temp+=Scores.compute_precision(tp,fp,zero_division)
            return temp/y_true.shape[1]
        elif average == 'micro':
            tp_s = sum(d.get('tp', 0) for d in cm)
            fp_s = sum(d.get('fp', 0) for d in cm)
            fn_s = sum(d.get('fn', 0) for d in cm)
            tn_s = sum(d.get('tn', 0) for d in cm)
            return Scores.compute_precision(tp_s,fp_s)
        
    def Recall(y_true, y_pred, average='macro', zero_division=0):
        cm = Scores.compute_tp_tn_fn_fp(y_true, y_pred)
        if (average == 'macro'):
            temp = 0
            for index in range(len(cm)):
                tp = cm[index]['tp']
                fp = cm[index]['fp']
                fn = cm[index]['fn']
                tn = cm[index]['tn']
                temp+=Scores.compute_recall(tp,fn,zero_division)
            return temp/y_true.shape[1]
        elif average == 'micro':
            tp_s = sum(d.get('tp', 0) for d in cm)
            fp_s = sum(d.get('fp', 0) for d in cm)
            fn_s = sum(d.get('fn', 0) for d in cm)
            tn_s = sum(d.get('tn', 0) for d in cm)
            return Scores.compute_recall(tp_s,fn_s)
    
        
if __name__=='__main__': 
    y_true = np.array([
                [0,1,0],
                [0,1,1],
                [1,0,1],
                [0,0,1]])

    y_pred = np.array([
                [0,1,1],
                [0,1,1],
                [0,1,0],
                [0,0,0]])
    
    # cm = Scores.compute_tp_tn_fn_fp(y_true, y_pred)
    
    print('Precision macro_averaged zero_division = 1:', Scores.Precision(y_true, y_pred, average='macro', zero_division=1))
    print('Precision sklearn macro zero_division = 1: {0}'.format(precision_score(y_true, y_pred, average='macro', zero_division=1))) 
    print('*****************************')
    
    print('Precision macro_averaged zero_division = 0:', Scores.Precision(y_true, y_pred, average='macro', zero_division=0))
    print('Precision sklearn macro zero_division = 0: {0}'.format(precision_score(y_true, y_pred, average='macro', zero_division=0))) 
    print('*****************************')
    
    print('Precision micro_averaged: ', Scores.Precision(y_true, y_pred, average='micro'))
    print('Precision sklearn micro_averaged: {0} '.format(precision_score(y_true, y_pred, average='micro'))) 
    print('*****************************')
    
    print('Recall macro_averaged zero_division = 1: ', Scores.Recall(y_true, y_pred, average='macro', zero_division=1))
    print('Recall sklearn macro zero_division = 1: {0}'.format(recall_score(y_true, y_pred, average='macro', zero_division=1))) 
    print('*****************************')
    
    print('Recall macro_averaged zero_division = 0:', Scores.Recall(y_true, y_pred, average='macro', zero_division=0))
    print('Recall sklearn macro zero_division = 0: {0}'.format(recall_score(y_true, y_pred, average='macro', zero_division=0))) 
    print('*****************************')
    
    print('Recall micro_averaged: ', Scores.Recall(y_true, y_pred, average='micro'))
    print('Recall sklearn micro_averaged: {0} '.format(recall_score(y_true, y_pred, average='micro'))) 
    print('*****************************')
    