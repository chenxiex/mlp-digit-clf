import warnings
import argparse
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from joblib import dump,load

# 定义训练函数
def train(model,train_data):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=ConvergenceWarning,module="sklearn")
        model.fit(train_data['X'],train_data['y'])

# 定义测试函数
def test(model,test_data):
    test_loss=log_loss(test_data['y'],model.predict_proba(test_data['X']))
    for data,target in test_data['X'],test_data['y']:
        output=model.predict(data)
        if target==output:
            correct+=1
        len=len(test_data['X'])
    print('测试结果：平均损失函数值：{:.4f},正确率：{}/{} ({:.0f}%)'.format(test_loss,correct,len,100.*correct/len))

def main():
    # 测试参数
    parser=argparse.ArgumentParser(description="MNIST MLP分类器")
    parser.add_argument('--batch-size',type=int,default=64,metavar='N',help='输入批次大小，默认为64')
    parser.add_argument('--test-batch-size',type=int,default=1000,metavar='N',help='测试批次大小，默认为1000')
    parser.add_argument('--max-iter',type=int,default=14,metavar='N',help='最大训练轮数，默认为14')
    parser.add_argument('--lr',type=float,default=1.0,metavar='LR',help='学习率，默认为1.0')
    parser.add_argument('--gamma',type=float,default=0.7,metavar='M',help='学习率衰减率，默认为0.7')
    parser.add_argument('--seed',type=int,default=1,metavar='S',help='随机种子，默认为1')
    parser.add_argument('--save-model',action='store_true',default=False,help='是否保存模型，默认为False')
    args=parser.parse_args()

    X,y=fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False,parser="pandas")
    # transform
    transform=StandartScaler()
    X=transform.fit_transform(X)
    # 划分测试与训练集
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=args.test_batch_size)

    model=MLPClassifier(hidden_layer_sizes=(40,),max_iter=args.max_iter,solver='sgd',batch_size=args.batch_size,learning_rate='invscaling',learning_rate_init=args.lr,power_t=args.gamma,random_state=args.seed,verbose=10)

    train(args,model,{'X':X_train,'y':y_train})
    test(model,{'X':X_test,'y':y_test})

    if args.save_model:
        dump(model,'mlp-mnist.joblib')


if __name__ == '__main__':
    main()