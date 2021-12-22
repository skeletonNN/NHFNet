import os
import datetime
import argparse
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from cmu_mosei import CMUMOSEIDataset
from networks import *
from main_utils import train, validation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# fixed seed
seed = 20
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

modalities = ('visual', 'audio', 'bert')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


# define model input
def get_X(device, sample):
    ret = []
    for m in modalities:
        X = sample[m].to(device)
        ret.append(X.float())
    n = ret[0].size(0)
    return ret, n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory', default='datasets')
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--num_workers', type=int, help='num workers', default=4)
    parser.add_argument('--epochs', type=int, help='train epochs', default=10)
    parser.add_argument('--checkpoint', type=str, help='model checkpoint for evaluation', default='')
    parser.add_argument('--checkpointdir', type=str, help='directory to save weights', default='checkpoints')
    parser.add_argument('--no_verbose', action='store_true', default=False, help='turn off verbose for training')
    parser.add_argument('--log_interval', type=int, help='interval for displaying training info if verbose', default=10)
    parser.add_argument('--no_save', action='store_true', default=False, help='set to not save model weights')
    parser.add_argument('--train', action='store_true', default=False, help='training')

    args = parser.parse_args()

    print("The configuration of this run is:")
    print(args, end='\n\n')

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device('cuda' if use_cuda else 'cpu')  # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': False} \
        if use_cuda else {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}

    # dataset folders
    training_folder = os.path.join(args.datadir, 'train')
    val_folder = os.path.join(args.datadir, 'val')
    test_folder = os.path.join(args.datadir, 'test')

    # Generators
    dataset_params = {
        'label': 'sentiment'
    }

    for m in modalities:
        dataset_params.update({m: None})

    # Load dataset
    training_set = CMUMOSEIDataset(training_folder, dataset_params)
    training_loader = data.DataLoader(training_set, **params)
    val_set = CMUMOSEIDataset(val_folder, dataset_params)
    val_loader = data.DataLoader(val_set, **params)
    test_set = CMUMOSEIDataset(test_folder, dataset_params)
    test_loader = data.DataLoader(test_set, **params)

    # define model
    model_param = {}
    if 'visual' in modalities:
        model = FACETVisualLSTMNet()
        print('Initialized model for video modality')
        model_param.update(
            {'visual': {
                'model': model,
                'id': modalities.index('visual')
            }})
    if 'audio' in modalities:
        model = COVAREPAudioLSTMNet()
        print('Initialized model for audio modality')
        model_param.update(
            {'audio': {
                'model': model,
                'id': modalities.index('audio')
            }})
    if 'bert' in modalities:
        model = BERTTextLSTMNet()
        print('Initialized model for bert')
        model_param.update(
            {'bert': {
                'model': model,
                'id': modalities.index('bert')
            }})
    multimodal_model = MSAFLSTMNet(model_param)
    print(get_n_params(multimodal_model))
    multimodal_model.to(device)

    # loss functions
    train_loss_func = torch.nn.MSELoss()
    val_loss_func = torch.nn.L1Loss()

    # train mode or eval mode
    if args.train:
        print('Training...')
        # Adam parameters
        optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=args.lr)

        # record training process
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = os.path.join(args.checkpointdir, 'logs/{}'.format(current_time))
        writer = SummaryWriter(log_dir=train_log_dir)
        test = []

        for epoch in range(args.epochs):
            # train, test model
            train_loss, epoch_train_scores = train(get_X, args.log_interval, multimodal_model, device, training_loader,
                                                   optimizer, train_loss_func, epoch, not args.no_verbose)
            epoch_test_loss, epoch_test_score = validation(get_X, multimodal_model, device, val_loss_func, val_loader)
            epoch_test_loss, epoch_test_score = validation(get_X, multimodal_model, device, val_loss_func, test_loader)

            if not args.no_save:
                states = {
                    'epoch': epoch + 1,
                    'model_state_dict': multimodal_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_score': epoch_test_score,
                    'test_loss': epoch_test_loss
                }
                torch.save(states, os.path.join(args.checkpointdir, 'msaf_mosei_epoch{}.pth'.format(epoch + 1)))
                print('Epoch {} model saved!'.format(epoch + 1))

            # save results
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', epoch_test_loss, epoch)
            writer.add_scalar('Acc7/train', epoch_train_scores[0], epoch)
            writer.add_scalar('Acc7/test', epoch_test_score[0], epoch)
            writer.add_scalar('Acc2/train', epoch_train_scores[1], epoch)
            writer.add_scalar('Acc2/test', epoch_test_score[1], epoch)
            writer.add_scalar('F1/train', epoch_train_scores[2], epoch)
            writer.add_scalar('F1/test', epoch_test_score[2], epoch)

            test.append(epoch_test_score)
            writer.flush()

        test = np.array(test)
        labels = ['Acc 7', 'Acc 2', 'F1', 'Corr']
        for scores, label in zip(test.T, labels):
            print('Best {} score {:.2f}% at epoch {}'.format(label, np.max(scores), np.argmax(scores)+1))

    else:
        if args.checkpoint:
            print('Evaluating...')
            model_path = args.checkpoint
            checkpoint = torch.load(model_path) if use_cuda else torch.load(model_path,
                                                                            map_location=torch.device('cpu'))
            multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            print('Loaded model from', model_path)
            test_set = CMUMOSEIDataset(test_folder, dataset_params)
            test_loader = data.DataLoader(test_set, **params)
            epoch_test_loss, epoch_test_score = validation(get_X, multimodal_model, device, val_loss_func, test_loader,
                                                           print_cm=True)
        else:
            print('--checkpoint not specified')
