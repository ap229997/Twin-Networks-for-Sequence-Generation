from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle
from itertools import chain

import opts
import models
from models.CaptionModel import *
from models.ShowTellModel import *
from dataloader import *
import eval_utils
import misc.utils as utils
import time

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    # log information
    folder_id='log_result'
    file_id='twin_show_attend_tell'
    log_file_name=os.path.join(folder_id,file_id + '.txt')
    log_file=open(log_file_name,'w')

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt)
    model.cuda()
    back_model = models.setup(opt, reverse=True) # True for twin-net
    back_model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()
    back_model.train()

    crit = utils.LanguageModelCriterion() # define the loss criterion
    all_param = chain(model.parameters(), back_model.parameters()) 
    optimizer = optim.Adam(all_param, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()
        
        # flip the masks and labels for twin-net
        reverse_labels = np.flip(data['labels'],1).copy()
        reverse_masks = np.flip(data['masks'], 1).copy()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'],reverse_labels, data['masks'], reverse_masks]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, reverse_labels, masks, reverse_masks = tmp

        optimizer.zero_grad()
        out, states = model(fc_feats, att_feats, labels)
        back_out, back_states = back_model(fc_feats, att_feats, reverse_labels)
        idx = [i for i in range(back_states.size()[1] - 1, -1, -1)]
        # print (back_states.size(), back_states.size()[1])
        # print (type(idx))
        # print (idx)

        idx = torch.LongTensor(idx)
        idx = Variable(idx).cuda()
        invert_backstates = back_states.index_select(1, idx)

        # print (states.size(), back_states.size())

        # check if the back states are inverted
        # back = back_states.index_select(1, Variable(torch.LongTensor([2])).cuda())
        # forw = invert_backstates.index_select(1, Variable(torch.LongTensor([14])).cuda())
        # print (forw, back)
        # print (back_states.index_select(1, Variable(torch.LongTensor([3])).cuda()))
        # print (invert_backstates.size())

        loss = crit(out, labels[:,1:], masks[:,1:]) # compute using the defined criterion
        
        back_loss = crit(back_out, reverse_labels[:,:-1], reverse_masks[:,:-1]) 
        
        invert_backstates = invert_backstates.detach()
        l2_loss = ((states - invert_backstates )** 2).mean()
        
        all_loss = loss + 1.5 * l2_loss + back_loss
        
        all_loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        # store the relevant values
        train_l2_loss = l2_loss.data[0]
        train_loss = loss.data[0]
        train_all_loss = all_loss.data[0]
        train_back_loss = back_loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, l2_loss = {:.3f}, back_loss = {:.3f}, all_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, train_l2_loss, train_back_loss, train_all_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'l2_loss', train_l2_loss, iteration)
                add_summary_value(tf_summary_writer, 'all_loss', train_all_loss, iteration)
                add_summary_value(tf_summary_writer, 'back_loss', train_back_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            log_line = 'Epoch [%d], Step [%d], all loss: %f,back_loss %f,train_l2_loss %f, train_loss %f, time %f ' % (
                    epoch,iteration,train_all_loss,train_back_loss,train_l2_loss,
                    train_loss,time.clock()
                    )
            log_file.write(log_line + '\n')

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k,v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
