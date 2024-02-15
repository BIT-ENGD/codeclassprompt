from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
import warnings
warnings.filterwarnings("ignore")

#https://blog.csdn.net/tujituji1007/article/details/115027803
class MCMetrics(object):
    def __init__(self,y_test, y_pre,classlist=None,type="macro"): # 多分类用macro
        if len(y_test) != len(y_pre):
            raise ValueError("The length of actual_label must be same as gt_label.")
        if type != "macro" and type != "micro" and type is not None:
            raise ValueError("type must be eigher macro or micro.")
        self.y_test=y_test
        self.y_pre=y_pre
        self.type=type
        self.classlist=classlist
        
    def f1_score(self,type=None):
        if type is None:
            type=self.type
    
        return f1_score(self.y_test,  self.y_pre,average=type) #多分类求f1值

    def confuison_matrioc(self):
        return confusion_matrix( self.y_test, self.y_pre) #混淆矩阵
        
    def precision(self,type=None):
        if type is None:
            type=self.type
        return precision_score(self.y_test,  self.y_pre,average=type) #多分类求precision
    
    def recall(self,type=None):
        if type is None:
            type=self.type
        return recall_score(self.y_test, self.y_pre,average=type)    #多分类求recall
    
    def show_report(self,type=None): # None or "macro"
        report =""
        all_class=list(set(self.y_test))
        classnum=[]
        for cls in range(len(all_class)):      
            classnum.append(self.y_test.count(all_class[cls]))
        
        f1=self.f1_score(type)
        precision=self.precision(type)
        recall =self.recall(type)
        
        report+=("-"*90+"\n")
        report+="class\tPrecision\tRecall\tF1\tSupport\n"
        for id,item in enumerate(zip(self.classlist,f1,recall,precision,classnum)):
            classid,f1_v,recall_v,precision_v,support_v=item
            report+="%s\t%8.5f\t%8.5f\t%8.5f\t%d\n"%(classid,precision_v,recall_v,f1_v,support_v)
        #total
        f1=self.f1_score("micro")
        precision=self.precision("micro")
        recall =self.recall("micro")
        report+=("-"*90+"\n")
        report+=("%s\t\t%8.5f\t%8.5f\t%8.5f\t%d \n"%("total",precision,recall,f1,len(self.y_pre)))
        report+=("-"*90+"\n")

        return report

    
def metrics_calculate(y_test,y_pre,label_data,show_detail=False,type="macro"):
    if( len (y_test) != len(y_pre)):
        raise ValueError("y_test and y_pre must have same number!")
    metricsobj=MCMetrics(y_test,y_pre,label_data,None)
    # f1=metricsobj.f1_score()

    # recall=metricsobj.recall()
    # print(recall)
    # precision=metricsobj.precision()
    # print(precision)

    # print("\n[ f1_score is %9.6f ]"%(f1))
    if show_detail:
        return metricsobj.show_report()
    else:
        f1=metricsobj.f1_score(type)
        recall=metricsobj.recall(type)
        precision=metricsobj.precision(type)
        return ("[F1_Score: %8.5f, Recall: %8.5f, Precision: %8.5f ]"%(f1,recall,precision))

    
def metrics_calculate_ex(y_test,y_pre,label_data,show_detail=False,type="micro"):
    if( len (y_test) != len(y_pre)):
        raise ValueError("y_test and y_pre must have same number!")
    metricsobj=MCMetrics(y_test,y_pre,label_data,None)
    # f1=metricsobj.f1_score()

    # recall=metricsobj.recall()
    # print(recall)
    # precision=metricsobj.precision()
    # print(precision)

    # print("\n[ f1_score is %9.6f ]"%(f1))
    f1=metricsobj.f1_score(type)
    recall=metricsobj.recall(type)
    precision=metricsobj.precision(type)
    return f1,recall,precision