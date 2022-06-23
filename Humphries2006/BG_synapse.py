from typing import Union, Dict, Callable

import brainpy as bp
import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq
from brainpy.types import Tensor

class BG_AMPA(bp.dyn.ExpCUBA):
    def __init__(self, 
        pre: NeuGroup, 
        post: NeuGroup, 
        conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]], 
        conn_type: str = 'sparse', 
        g_max: Union[float, Tensor, Initializer, Callable] = 1, 
        delay_step: Union[int, Tensor, Initializer, Callable] = None, 
        tau: Union[float, Tensor] = 8, 
        name: str = None, 
        method: str = 'exp_auto'
        ):
        super().__init__(pre, 
        post, 
        conn, 
        conn_type, 
        g_max, 
        delay_step, 
        tau,
        name, 
        method)

        self.check_pre_attrs('spike')
        self.check_post_attrs('I_AMPA', 'V')

        # parameters
        self.tau = tau
        if bm.size(self.tau) != 1:
            raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. '
                        f'But we got {self.tau}')

        # connections and weights
        self.conn_type = conn_type
        if conn_type not in ['sparse', 'dense']:
            raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
        if self.conn is None:
            raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
        if isinstance(self.conn, One2One):
            self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
            self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        elif isinstance(self.conn, All2All):
            self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
            if bm.size(self.g_max) != 1:
                self.weight_type = 'heter'
                bm.fill_diagonal(self.g_max, 0.)
            else:
                self.weight_type = 'homo'
        else:
            if conn_type == 'sparse':
                self.pre2post = self.conn.require('pre2post')
                self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
            elif conn_type == 'dense':
                self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
                if self.weight_type == 'homo':
                    self.conn_mat = self.conn.require('conn_mat')
            else:
                raise ValueError(f'Unknown connection type: {conn_type}')

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                            delay_step,
                                            self.pre.spike)

        # function
        self.integral = odeint(lambda g, t: -g / self.tau, method=method)


    def reset(self):
        self.g.value = bm.zeros(self.post.num)

    def update(self, t, dt):
        # delays
        pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

        # post values
        assert self.weight_type in ['homo', 'heter']
        assert self.conn_type in ['sparse', 'dense']
        if isinstance(self.conn, All2All):
            pre_spike = pre_spike.astype(bm.float_)
            if self.weight_type == 'homo':
                post_vs = bm.sum(pre_spike)
                if not self.conn.include_self:
                    post_vs = post_vs - pre_spike
                post_vs = self.g_max * post_vs
            else:
                post_vs = pre_spike @ self.g_max
        elif isinstance(self.conn, One2One):
            pre_spike = pre_spike.astype(bm.float_)
            post_vs = pre_spike * self.g_max
        else:
            if self.conn_type == 'sparse':
                post_vs = bm.pre2post_event_sum(pre_spike,
                                            self.pre2post,
                                            self.post.num,
                                            self.g_max)
            else:
                pre_spike = pre_spike.astype(bm.float_)
                if self.weight_type == 'homo':
                    post_vs = self.g_max * (pre_spike @ self.conn_mat)
                else:
                    post_vs = pre_spike @ self.g_max

        # updates
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        self.post.I_AMPA += self.output2(self.g)

    def output2(self,g_post):
        return g_post


class BG_NMDA(bp.dyn.ExpCUBA):
    def __init__(self, 
        pre: NeuGroup, 
        post: NeuGroup, 
        conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]], 
        conn_type: str = 'sparse', 
        g_max: Union[float, Tensor, Initializer, Callable] = 1, 
        delay_step: Union[int, Tensor, Initializer, Callable] = None, 
        tau: Union[float, Tensor] = 8, 
        name: str = None, 
        method: str = 'exp_auto'
        ):
        super().__init__(pre, 
        post, 
        conn, 
        conn_type, 
        g_max, 
        delay_step, 
        tau,
        name, 
        method)

        self.check_pre_attrs('spike')
        self.check_post_attrs('I_NMDA', 'V')

        # parameters
        self.tau = tau
        if bm.size(self.tau) != 1:
            raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. '
                        f'But we got {self.tau}')

        # connections and weights
        self.conn_type = conn_type
        if conn_type not in ['sparse', 'dense']:
            raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
        if self.conn is None:
            raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
        if isinstance(self.conn, One2One):
            self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
            self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        elif isinstance(self.conn, All2All):
            self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
            if bm.size(self.g_max) != 1:
                self.weight_type = 'heter'
                bm.fill_diagonal(self.g_max, 0.)
            else:
                self.weight_type = 'homo'
        else:
            if conn_type == 'sparse':
                self.pre2post = self.conn.require('pre2post')
                self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
            elif conn_type == 'dense':
                self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
                if self.weight_type == 'homo':
                    self.conn_mat = self.conn.require('conn_mat')
            else:
                raise ValueError(f'Unknown connection type: {conn_type}')

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                            delay_step,
                                            self.pre.spike)

        # function
        self.integral = odeint(lambda g, t: -g / self.tau, method=method)


    def reset(self):
        self.g.value = bm.zeros(self.post.num)

    def update(self, t, dt):
        # delays
        pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

        # post values
        assert self.weight_type in ['homo', 'heter']
        assert self.conn_type in ['sparse', 'dense']
        if isinstance(self.conn, All2All):
            pre_spike = pre_spike.astype(bm.float_)
            if self.weight_type == 'homo':
                post_vs = bm.sum(pre_spike)
                if not self.conn.include_self:
                    post_vs = post_vs - pre_spike
                post_vs = self.g_max * post_vs
            else:
                post_vs = pre_spike @ self.g_max
        elif isinstance(self.conn, One2One):
            pre_spike = pre_spike.astype(bm.float_)
            post_vs = pre_spike * self.g_max
        else:
            if self.conn_type == 'sparse':
                post_vs = bm.pre2post_event_sum(pre_spike,
                                            self.pre2post,
                                            self.post.num,
                                            self.g_max)
            else:
                pre_spike = pre_spike.astype(bm.float_)
                if self.weight_type == 'homo':
                    post_vs = self.g_max * (pre_spike @ self.conn_mat)
                else:
                    post_vs = pre_spike @ self.g_max

        # updates
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        self.post.I_NMDA += self.output2(self.g)

    def output2(self,g_post):
        return g_post



class BG_GABA(bp.dyn.ExpCUBA):
    def __init__(self, 
        pre: NeuGroup, 
        post: NeuGroup, 
        conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]], 
        conn_type: str = 'sparse', 
        g_max: Union[float, Tensor, Initializer, Callable] = 1, 
        delay_step: Union[int, Tensor, Initializer, Callable] = None, 
        tau: Union[float, Tensor] = 8, 
        name: str = None, 
        method: str = 'exp_auto',
        proportion=bm.Variable([0.33,0.33,0.34])
        ):
        super().__init__(pre, 
        post, 
        conn, 
        conn_type, 
        g_max, 
        delay_step, 
        tau,
        name, 
        method)

        self.check_pre_attrs('spike')
        self.check_post_attrs('I_GABA_D','I_GABA_P','I_GABA_S', 'V')

        self.proportion = proportion#self.post.proportion S P D

        random_matrix_tmp = bm.random.uniform(size=self.post.num)
        c_1 = self.proportion[0]
        c_2 = self.proportion[0]+self.proportion[1]
        self.mask_S = random_matrix_tmp<c_1
        #self.mask_P = bm.where(c_1<random_matrix_tmp<c_2)
        self.mask_D = c_2<random_matrix_tmp
        
        self.mask_P = (1-self.mask_D-self.mask_S).astype(bm.bool_)

        # parameters
        self.tau = tau
        if bm.size(self.tau) != 1:
            raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. '
                        f'But we got {self.tau}')

        # connections and weights
        self.conn_type = conn_type
        if conn_type not in ['sparse', 'dense']:
            raise ValueError(f'"conn_type" must be in "sparse" and "dense", but we got {conn_type}')
        if self.conn is None:
            raise ValueError(f'Must provide "conn" when initialize the model {self.name}')
        if isinstance(self.conn, One2One):
            self.g_max = init_param(g_max, (self.pre.num,), allow_none=False)
            self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
        elif isinstance(self.conn, All2All):
            self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
            if bm.size(self.g_max) != 1:
                self.weight_type = 'heter'
                bm.fill_diagonal(self.g_max, 0.)
            else:
                self.weight_type = 'homo'
        else:
            if conn_type == 'sparse':
                self.pre2post = self.conn.require('pre2post')
                self.g_max = init_param(g_max, self.pre2post[1].shape, allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
            elif conn_type == 'dense':
                self.g_max = init_param(g_max, (self.pre.num, self.post.num), allow_none=False)
                self.weight_type = 'heter' if bm.size(self.g_max) != 1 else 'homo'
                if self.weight_type == 'homo':
                    self.conn_mat = self.conn.require('conn_mat')
            else:
                raise ValueError(f'Unknown connection type: {conn_type}')

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay_step = self.register_delay(f"{self.pre.name}.spike",
                                            delay_step,
                                            self.pre.spike)

        # function
        self.integral = odeint(lambda g, t: -g / self.tau, method=method)


    def reset(self):
        self.g.value = bm.zeros(self.post.num)

    def update(self, t, dt):
        # delays
        pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)

        # post values
        assert self.weight_type in ['homo', 'heter']
        assert self.conn_type in ['sparse', 'dense']
        if isinstance(self.conn, All2All):
            pre_spike = pre_spike.astype(bm.float_)
            if self.weight_type == 'homo':
                post_vs = bm.sum(pre_spike)
                if not self.conn.include_self:
                    post_vs = post_vs - pre_spike
                post_vs = self.g_max * post_vs
            else:
                post_vs = pre_spike @ self.g_max
        elif isinstance(self.conn, One2One):
            pre_spike = pre_spike.astype(bm.float_)
            post_vs = pre_spike * self.g_max
        else:
            if self.conn_type == 'sparse':
                post_vs = bm.pre2post_event_sum(pre_spike,
                                            self.pre2post,
                                            self.post.num,
                                            self.g_max)
            else:
                pre_spike = pre_spike.astype(bm.float_)
                if self.weight_type == 'homo':
                    post_vs = self.g_max * (pre_spike @ self.conn_mat)
                else:
                    post_vs = pre_spike @ self.g_max

        # updates
        self.g.value = self.integral(self.g.value, t, dt=dt) + post_vs
        out = self.output2(self.g)
        self.post.I_GABA_D -= out[0]
        self.post.I_GABA_P -= out[1]
        self.post.I_GABA_S -= out[2]


    def output2(self, g_post):
        z = bm.zeros_like(g_post)
        g_D = bm.where(self.mask_D,g_post,z)
        g_P = bm.where(self.mask_P,g_post,z)
        g_S = bm.where(self.mask_S,g_post,z)
        return [g_D,g_P,g_S]