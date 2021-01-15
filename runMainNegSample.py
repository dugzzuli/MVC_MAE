import numpy as np
import linecache
from Dataset.dataset import Dataset
from Model.mModel import MVModel
from Trainer.trainerNegSample import  TrainerNegSample as Trainer #
from Trainer.pretrainer import PreTrainer
import os
import random
import tensorflow as tf
import yaml
import argparse
import warnings

from Utils import gpu_info
from Utils.utils import mkdir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)



if __name__=='__main__':
    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print(gpus_to_use, free_memory)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser(description='Deep subspace')
    parser.add_argument('--dataset', default='NUSWIDEOBJ', type=str, help='dataset')
    args = parser.parse_args()
    print(args)

    d_a = yaml.load(open("configSample.yaml", 'r'))

    dataset_name=args.dataset.replace('\n', '').replace('\r', '')
    View_num= d_a[dataset_name]['View_num']
    layers = d_a[dataset_name]['layers']
    random.seed(9001)

    dataset_config = {
        'View': ['./Database/' + dataset_name + '/View{}.txt'.format(i + 1) for i in range(View_num)],
        'W': ['./Database/' + dataset_name + '/W{}.txt'.format(i + 1) for i in range(View_num)],
        'label_file': './Database/' + dataset_name + '/group.txt'}
    graph = Dataset(dataset_config,loadW=d_a[dataset_name]['loadW'])

    dims = [np.shape(vData)[1] for vData in graph.ViewData]


    graph.normData()

    pretrain_config = {
        'iter':30000,
        'View':layers,
        'pretrain_params_path': './Log/'+dataset_name+'/pretrain_params.pkl'}


    if d_a[dataset_name]['pretrain']:
        pretrainer = PreTrainer(pretrain_config,learning_rate=1e-3)

        for i in range(len(graph.ViewData)):
            pretrainer.pretrain(graph.ViewData[i], 'V'+str(i+1))

    model_config = {
        'weight_decay': 0.00001,
        'View_num':View_num,
        'View': layers,
        'is_init': True,
        'pretrain_params_path': './Log/'+dataset_name+'/pretrain_params.pkl'
    }

    # L = 100.0000 & beta_i = 10.0000 & alpha_i = 0.0000 & gama_i = 200.0000 & acc = 0.9319 & nmi = 0.9693

    # beta_W=d_a[dataset_name]['beta_W']

    for learning_rate in d_a[dataset_name]['learning_rate'][::-1]:
        dirP = './result/MainSample/{}/{}/{}/'.format( d_a['resultDirName'],dataset_name, learning_rate)
        mkdir(dirP)

        with open(dirP + 'data_each_layer_gama.txt', 'w') as f:
            for beta_W in d_a[dataset_name]['beta_W']: #dec 损失
                beta_W=[beta_W]*View_num
                for L in d_a[dataset_name]['L']: #dec 损失
                    for beta_i in [1]:
                        for alpha_i in d_a[dataset_name]['alpha']:
                            for gama_i in d_a[dataset_name]['gama']: #first_order_loss
                                tf.reset_default_graph()
                                trainer_config = {
                                    'loadW' : d_a[dataset_name]['loadW'],
                                    'beta_W':beta_W,
                                    'dims':dims,
                                    'View': layers,
                                    'drop_prob': 0.2,
                                    'learning_rate': learning_rate,
                                    'batch_size': d_a[dataset_name]['batch_size'],
                                    'num_epochs': d_a[dataset_name]['ft_times'],
                                    'beta': beta_i,
                                    'alpha': alpha_i,
                                    'gamma': gama_i,
                                    'L': L,
                                    'sent_outputs_norm': d_a[dataset_name]['sent_outputs_norm'],
                                    'cluster_num': graph.num_classes,
                                    'View_num': View_num,
                                    'model_path': './Log/'+dataset_name+'/'+dataset_name+'_model.pkl',
                                }

                                model = MVModel(model_config)
                                trainer = Trainer(model, trainer_config)
                                accList,nmiList=trainer.train(graph)
                                result_single = 'max_acc={:.4f}'.format(np.max(accList)) + ' & ' + 'max_nmi={:.4f}'.format(np.max(nmiList))
                                f.write(result_single + '\n')

                                acc,nmi=trainer.inferClusterDug(graph)

                                result_single =  'L={:.4f}'.format(L) + ' & beta_i={:.4f}'.format(beta_i) + ' & alpha_i={:.4f}'.format(alpha_i) + ' & gama_i={:.4f}'.format(gama_i) + ' & acc={:.4f}'.format(
                                                acc) + ' & ' + 'nmi={:.4f}'.format(nmi)
                                f.write(result_single + '\n')
                                f.flush()

                                trainer.save_model()

