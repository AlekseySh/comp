from fastai.callbacks.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.tabular import *
from matplotlib import pyplot as plt
from torch import tensor
from torch.nn import CrossEntropyLoss as CEloss

import src.data_utils as u


def sigmoid_focal_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        weight,
        gamma: float = 2.0,
        alpha: float = 1.0,
        reduction: str = "mean"
):
    targets = targets.type(outputs.type())
    targets = torch.stack((targets < torch.tensor(0.5).cuda(),
                           targets > torch.tensor(0.5).cuda())
                          ).float().transpose(0, 1)

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none", weight=weight
    )
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def fit_predict_cv(train: pd.DataFrame,
                   test: pd.DataFrame,
                   val_ids: List[np.ndarray],
                   cat_cols: List[str],
                   cont_cols: List[str],
                   draw_plot: bool
                   ) -> Tuple[np.ndarray, List[float], List[float]]:
    u.random_seed(42)

    device_ = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')

    p = {
        'bs': 200_000,
        'n_epochs': 20,
        'layers': [512, 256, 128],
        'weights': [1, 10],
        'n_steps_f1': 20,
        'emb_drop': 0.5
    }

    procs = [FillMissing, Categorify, Normalize]

    test_tab = TabularList.from_df(df=test, cat_names=cat_cols,
                                   cont_names=cont_cols)

    k_fold = len(val_ids)
    probas_test = np.zeros((k_fold, len(test)), dtype=np.float16)

    work_dir = '../results/fai/'  # + str(datetime.datetime.now()).replace(' ', '')
    Path(work_dir).mkdir(exist_ok=True, parents=True)

    with open(Path(work_dir) / 'parameters.json', 'w') as f:
        json.dump(p, f)

    val_scores, ths = [], []
    for k, cur_val_ids in enumerate(val_ids):
        print(f'Fold {1 + k} / {k_fold} start.')

        data_ = (TabularList.from_df(
            train, procs=procs, cat_names=cat_cols, cont_names=cont_cols)
                 .split_by_idx(cur_val_ids)
                 .label_from_df(cols='y')
                 .add_test(test_tab)
                 .databunch(bs=p['bs']))

        learn = tabular_learner(data_, path=work_dir, layers=p['layers'],
                                emb_drop=p['emb_drop'],
                                metrics=u.F1(0, 1, steps=p['n_steps_f1']),
                                callback_fns=[ShowGraph,
                                              partial(EarlyStoppingCallback,
                                                      monitor='f1',
                                                      min_delta=0.001,
                                                      patience=7)
                                              ],
                                loss_func=CEloss(
                                    weight=tensor(p['weights']).float().to(device_)
                                ),
                                opt_func=torch.optim.Adam
                                )
        learn.lr_find()

        if draw_plot:
            learn.recorder.plot()
            plt.show()

        learn.fit_one_cycle(p['n_epochs'], max_lr=slice(5e-3),
                            callbacks=[
                                SaveModelCallback(learn, every='improvement',
                                                  monitor='f1', name=f'best{k}')
                            ])

        if draw_plot:
            learn.recorder.plot_losses()
            plt.show()
            learn.recorder.plot_lr()
            plt.show()

        # Test
        probas, *_ = learn.get_preds(DatasetType.Test)
        probas_test[k, :] = probas[:, 1]

        # Valid
        probas_val, *_ = learn.get_preds(DatasetType.Valid)
        score, th = u.f1_flexible(gts=train['y'].values[cur_val_ids],
                                  probas=probas_val[:, 1],
                                  th_start=0, th_stop=1,
                                  steps=p['n_steps_f1'])
        val_scores.append(score)
        ths.append(th)

        del learn, data_

    print(f'CV mean score is {np.mean(val_scores)}.')
    return probas_test, ths, val_scores


def get_max_from_log(learn: Learner,
                     metric_name: str = 'f1'
                     ) -> float:
    i_metric = learn.metrics_names.index(metric_name)
    metrics_log = learn.metrics[i_metric]

    i_max = np.argmax(metrics_log)
    max_score = metrics_log[i_max]

    print(f'Best score arised on {i_max} epoch: {max_score}.')

    return max_score
