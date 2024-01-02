'''
///////////////////////////////////////
3D LiDAR Object Detection - ADAS
Pranav Durai
//////////////////////////////////////
'''
import torch
from torch.optim import SGD, lr_scheduler
import numpy as np


class _LRMomentumScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_momentum', group['momentum'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_momentum' not in group:
                    raise KeyError("param 'initial_momentum' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_momentums = list(map(lambda group: group['initial_momentum'], optimizer.param_groups))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def get_momentum(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr, momentum in zip(self.optimizer.param_groups, self.get_lr(), self.get_momentum()):
            param_group['lr'] = lr
            param_group['momentum'] = momentum


class ParameterUpdate(object):
    """A callable class used to define an arbitrary schedule defined by a list.
    This object is designed to be passed to the LambdaLR or LambdaScheduler scheduler to apply
    the given schedule.

    Arguments:
        params {list or numpy.array} -- List or numpy array defining parameter schedule.
        base_param {float} -- Parameter value used to initialize the optimizer.
    """

    def __init__(self, params, base_param):
        self.params = np.hstack([params, 0])
        self.base_param = base_param

    def __call__(self, epoch):
        return self.params[epoch] / self.base_param


def apply_lambda(last_epoch, bases, lambdas):
    return [base * lmbda(last_epoch) for lmbda, base in zip(lambdas, bases)]


class LambdaScheduler(_LRMomentumScheduler):
    """Sets the learning rate and momentum of each parameter group to the initial lr and momentum
    times a given function. When last_epoch=-1, sets initial lr and momentum to the optimizer
    values.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
            Default: lambda x:x.
        momentum_lambda (function or list): As for lr_lambda but applied to momentum.
            Default: lambda x:x.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lr_lambda = [
        ...     lambda epoch: epoch // 30,
        ...     lambda epoch: 0.95 ** epoch
        ... ]
        >>> mom_lambda = [
        ...     lambda epoch: max(0, (50 - epoch) // 50),
        ...     lambda epoch: 0.99 ** epoch
        ... ]
        >>> scheduler = LambdaScheduler(optimizer, lr_lambda, mom_lambda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda=lambda x: x, momentum_lambda=lambda x: x, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)

        if not isinstance(momentum_lambda, (list, tuple)):
            self.momentum_lambdas = [momentum_lambda] * len(optimizer.param_groups)
        else:
            if len(momentum_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} momentum_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(momentum_lambda)))
            self.momentum_lambdas = list(momentum_lambda)

        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate and momentum lambda functions will only be saved if they are
        callable objects and not if they are functions or lambdas.
        """
        state_dict = {key: value for key, value in self.__dict__.items()
                      if key not in ('optimizer', 'lr_lambdas', 'momentum_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)
        state_dict['momentum_lambdas'] = [None] * len(self.momentum_lambdas)

        for idx, (lr_fn, mom_fn) in enumerate(zip(self.lr_lambdas, self.momentum_lambdas)):
            if not isinstance(lr_fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = lr_fn.__dict__.copy()
            if not isinstance(mom_fn, types.FunctionType):
                state_dict['momentum_lambdas'][idx] = mom_fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop('lr_lambdas')
        momentum_lambdas = state_dict.pop('momentum_lambdas')
        self.__dict__.update(state_dict)

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

        for idx, fn in enumerate(momentum_lambdas):
            if fn is not None:
                self.momentum_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        return apply_lambda(self.last_epoch, self.base_lrs, self.lr_lambdas)

    def get_momentum(self):
        return apply_lambda(self.last_epoch, self.base_momentums, self.momentum_lambdas)


class ParameterUpdate(object):
    """A callable class used to define an arbitrary schedule defined by a list.
    This object is designed to be passed to the LambdaLR or LambdaScheduler scheduler to apply
    the given schedule. If a base_param is zero, no updates are applied.

    Arguments:
        params {list or numpy.array} -- List or numpy array defining parameter schedule.
        base_param {float} -- Parameter value used to initialize the optimizer.
    """

    def __init__(self, params, base_param):
        self.params = np.hstack([params, 0])
        self.base_param = base_param

        if base_param < 1e-12:
            self.base_param = 1
            self.params = self.params * 0.0 + 1.0

    def __call__(self, epoch):
        return self.params[epoch] / self.base_param


class ListScheduler(LambdaScheduler):
    """Sets the learning rate and momentum of each parameter group to values defined by lists.
    When last_epoch=-1, sets initial lr and momentum to the optimizer values. One of both of lr
    and momentum schedules may be specified.
    Note that the parameters used to initialize the optimizer are overriden by those defined by
    this scheduler.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lrs (list or numpy.ndarray): A list of learning rates, or a list of lists, one for each
            parameter group. One- or two-dimensional numpy arrays may also be passed.
        momentum (list or numpy.ndarray): A list of momentums, or a list of lists, one for each
            parameter group. One- or two-dimensional numpy arrays may also be passed.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lrs = [
        ...     np.linspace(0.01, 0.1, 100),
        ...     np.logspace(-2, 0, 100)
        ... ]
        >>> momentums = [
        ...     np.linspace(0.85, 0.95, 100),
        ...     np.linspace(0.8, 0.99, 100)
        ... ]
        >>> scheduler = ListScheduler(optimizer, lrs, momentums)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lrs=None, momentums=None, last_epoch=-1):
        groups = optimizer.param_groups
        if lrs is None:
            lr_lambda = lambda x: x
        else:
            lrs = np.array(lrs) if isinstance(lrs, (list, tuple)) else lrs
            if len(lrs.shape) == 1:
                lr_lambda = [ParameterUpdate(lrs, g['lr']) for g in groups]
            else:
                lr_lambda = [ParameterUpdate(l, g['lr']) for l, g in zip(lrs, groups)]

        if momentums is None:
            momentum_lambda = lambda x: x
        else:
            momentums = np.array(momentums) if isinstance(momentums, (list, tuple)) else momentums
            if len(momentums.shape) == 1:
                momentum_lambda = [ParameterUpdate(momentums, g['momentum']) for g in groups]
            else:
                momentum_lambda = [ParameterUpdate(l, g['momentum']) for l, g in zip(momentums, groups)]
        super().__init__(optimizer, lr_lambda, momentum_lambda)


class RangeFinder(ListScheduler):
    """Scheduler class that implements the LR range search specified in:
        A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch
        size, momentum, and weight decay. Leslie N. Smith, 2018, arXiv:1803.09820.

    Logarithmically spaced learning rates from 1e-7 to 1 are searched. The number of increments in
    that range is determined by 'epochs'.
    Note that the parameters used to initialize the optimizer are overriden by those defined by
    this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        epochs (int): Number of epochs over which to run test.
    Example:
        >>> scheduler = RangeFinder(optimizer, 100)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, epochs):
        lrs = np.logspace(-7, 0, epochs)
        super().__init__(optimizer, lrs)


class OneCyclePolicy(ListScheduler):
    """Scheduler class that implements the 1cycle policy search specified in:
        A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch
        size, momentum, and weight decay. Leslie N. Smith, 2018, arXiv:1803.09820.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr (float or list). Maximum learning rate in range. If a list of values is passed, they
            should correspond to parameter groups.
        epochs (int): The number of epochs to use during search.
        momentum_rng (list). Optional upper and lower momentum values (may be both equal). Set to
            None to run without momentum. Default: [0.85, 0.95]. If a list of lists is passed, they
            should correspond to parameter groups.
        phase_ratio (float): Fraction of epochs used for the increasing and decreasing phase of
            the schedule. For example, if phase_ratio=0.45 and epochs=100, the learning rate will
            increase from lr/10 to lr over 45 epochs, then decrease back to lr/10 over 45 epochs,
            then decrease to lr/100 over the remaining 10 epochs. Default: 0.45.
    """

    def __init__(self, optimizer, lr, epochs, momentum_rng=[0.85, 0.95], phase_ratio=0.45):
        phase_epochs = int(phase_ratio * epochs)
        if isinstance(lr, (list, tuple)):
            lrs = [
                np.hstack([
                    np.linspace(l * 1e-1, l, phase_epochs),
                    np.linspace(l, l * 1e-1, phase_epochs),
                    np.linspace(l * 1e-1, l * 1e-2, epochs - 2 * phase_epochs),
                ]) for l in lr
            ]
        else:
            lrs = np.hstack([
                np.linspace(lr * 1e-1, lr, phase_epochs),
                np.linspace(lr, lr * 1e-1, phase_epochs),
                np.linspace(lr * 1e-1, lr * 1e-2, epochs - 2 * phase_epochs),
            ])

        if momentum_rng is not None:
            momentum_rng = np.array(momentum_rng)
            if len(momentum_rng.shape) == 2:
                for i, g in enumerate(optimizer.param_groups):
                    g['momentum'] = momentum_rng[i][1]
                momentums = [
                    np.hstack([
                        np.linspace(m[1], m[0], phase_epochs),
                        np.linspace(m[0], m[1], phase_epochs),
                        np.linspace(m[1], m[1], epochs - 2 * phase_epochs),
                    ]) for m in momentum_rng
                ]
            else:
                for i, g in enumerate(optimizer.param_groups):
                    g['momentum'] = momentum_rng[1]
                momentums = np.hstack([
                    np.linspace(momentum_rng[1], momentum_rng[0], phase_epochs),
                    np.linspace(momentum_rng[0], momentum_rng[1], phase_epochs),
                    np.linspace(momentum_rng[1], momentum_rng[1], epochs - 2 * phase_epochs),
                ])
        else:
            momentums = None

        super().__init__(optimizer, lrs, momentums)
