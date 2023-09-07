import seaborn as sn
from sklearn.metrics import confusion_matrix

def fail_index(pred, real, cross_val_num):
    failed=[]
    with open('test_fold_index_2.pickle', 'rb') as f:
        test_fold_index = pickle.load(f)
    
    for i in range(len(pred)):
        if pred[i].item()!=real[i].detach().cpu().item():
            failed.append(test_fold_index[cross_val_num-1][i])
    
    return failed

def confusionmatrix(y_pred1, y_test1, column=['class0','class1','class2','class3','class4']):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    # _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    a=[]
    b=[]
    for i in range(len(y_test1)):
        a.append(y_test1[i].detach().cpu().item())
        b.append(y_pred1[i].item())
    y_test=a
    y_pred=b
    df_cm = pd.DataFrame(confusion_matrix(y_test,y_pred))
    df_cm.index=column
    df_cm.columns=column
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    b, t = plt.ylim() 
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)
    

    df_cm =df_cm / df_cm.astype(np.float).sum(axis=1)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    b, t = plt.ylim() 
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)

    