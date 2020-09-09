from itertools import chain
from logging import getLogger

import numpy as np


log = getLogger(__name__)


def _print_conll_report(results, accuracy, total_true_entities, total_predicted_entities, n_tokens, total_correct,
                        short_report=False, entity_of_interest=None):
    tags = list(results.keys())

    s = 'processed {len} tokens ' \
        'with {tot_true} phrases; ' \
        'found: {tot_pred} phrases;' \
        ' correct: {tot_cor}.\n\n'.format(len=n_tokens,
                                          tot_true=total_true_entities,
                                          tot_pred=total_predicted_entities,
                                          tot_cor=total_correct)

    s += 'precision:  {tot_prec:.2f}%; ' \
         'recall:  {tot_recall:.2f}%; ' \
         'FB1:  {tot_f1:.2f}\n\n'.format(acc=accuracy,
                                         tot_prec=results['__total__']['precision'],
                                         tot_recall=results['__total__']['recall'],
                                         tot_f1=results['__total__']['f1'])

    if not short_report:
        for tag in tags:
            if entity_of_interest is not None:
                if entity_of_interest in tag:
                    s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                      'recall:  {tot_recall:.2f}%; ' \
                                      'F1:  {tot_f1:.2f} ' \
                                      '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                   tot_recall=results[tag]['recall'],
                                                                   tot_f1=results[tag]['f1'],
                                                                   tot_predicted=results[tag]['n_pred'])
            elif tag != '__total__':
                s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                  'recall:  {tot_recall:.2f}%; ' \
                                  'F1:  {tot_f1:.2f} ' \
                                  '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                               tot_recall=results[tag]['recall'],
                                                               tot_f1=results[tag]['f1'],
                                                               tot_predicted=results[tag]['n_pred'])
    elif entity_of_interest is not None:
        s += '\t' + entity_of_interest + ': precision:  {tot_prec:.2f}%; ' \
                                         'recall:  {tot_recall:.2f}%; ' \
                                         'F1:  {tot_f1:.2f} ' \
                                         '{tot_predicted}\n\n'.format(tot_prec=results[entity_of_interest]['precision'],
                                                                      tot_recall=results[entity_of_interest]['recall'],
                                                                      tot_f1=results[entity_of_interest]['f1'],
                                                                      tot_predicted=results[entity_of_interest][
                                                                          'n_pred'])
    log.debug(s)


def _global_stats_f1(results):
    total_true_entities = 0
    total_predicted_entities = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_correct = 0
    for tag in results:
        if tag == '__total__':
            continue

        n_pred = results[tag]['n_pred']
        n_true = results[tag]['n_true']
        total_correct += results[tag]['tp']
        total_true_entities += n_true
        total_predicted_entities += n_pred
        total_precision += results[tag]['precision'] * n_pred
        total_recall += results[tag]['recall'] * n_true
        total_f1 += results[tag]['f1'] * n_true
    if total_true_entities > 0:
        accuracy = total_correct / total_true_entities * 100
        total_recall = total_recall / total_true_entities
    else:
        accuracy = 0
        total_recall = 0
    if total_predicted_entities > 0:
        total_precision = total_precision / total_predicted_entities
    else:
        total_precision = 0

    if total_precision + total_recall > 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0

    total_res = {'n_predicted_entities': total_predicted_entities,
                 'n_true_entities': total_true_entities,
                 'precision': total_precision,
                 'recall': total_recall,
                 'f1': total_f1}
    return total_res, accuracy, total_true_entities, total_predicted_entities, total_correct


def ner_token_f1(y_true, y_pred, print_results=False):
    y_true = list(chain(*y_true))
    y_pred = list(chain(*y_pred))

    # Drop BIO or BIOES markup
    assert all(len(str(tag).split('-')) <= 2 for tag in y_true)

    y_true = [str(tag).split('-')[-1] for tag in y_true]
    y_pred = [str(tag).split('-')[-1] for tag in y_pred]
    tags = set(y_true) | set(y_pred)
    tags_dict = {tag: n for n, tag in enumerate(tags)}

    y_true_inds = np.array([tags_dict[tag] for tag in y_true])
    y_pred_inds = np.array([tags_dict[tag] for tag in y_pred])

    results = {}
    for tag, tag_ind in tags_dict.items():
        if tag == 'O':
            continue
        tp = np.sum((y_true_inds == tag_ind) & (y_pred_inds == tag_ind))
        fn = np.sum((y_true_inds == tag_ind) & (y_pred_inds != tag_ind))
        fp = np.sum((y_true_inds != tag_ind) & (y_pred_inds == tag_ind))
        n_pred = np.sum(y_pred_inds == tag_ind)
        n_true = np.sum(y_true_inds == tag_ind)
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        results[tag] = {'precision': precision, 'recall': recall,
                        'f1': f1, 'n_true': n_true, 'n_pred': n_pred,
                        'tp': tp, 'fp': fp, 'fn': fn}

    results['__total__'], accuracy, total_true_entities, total_predicted_entities, total_correct = _global_stats_f1(
        results)
    n_tokens = len(y_true)
    if print_results:
        log.debug('TOKEN LEVEL F1')
        _print_conll_report(results, accuracy, total_true_entities, total_predicted_entities, n_tokens, total_correct)
    return results['__total__']['f1']
