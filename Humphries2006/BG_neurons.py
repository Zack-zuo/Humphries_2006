import brainpy as bp
import brainpy.math as bm


from typing import Union, Sequence

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.initialize import Initializer
from brainpy.types import Shape

import sys

sys.path.append('./')
from BGmodel import HumphriesNeuron


class D1Striatum(HumphriesNeuron):
    def __init__(self,
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=bp.init.ZeroInit(),
        noise=None,
        noise_type='value',
        keep_size=False,
        method='exp_auto',
        name=None,
        I_spon=0,
        V_lim=-20,
        lambda_1=0
    ):
        super().__init__(
            size,
            V_rest,
            V_th,
            R,
            tau,
            tau_ref,
            V_initializer,
            noise,
            noise_type,
            keep_size,
            method,
            name,
            I_spon,
            V_lim
        )

        self.lambda_1 = lambda_1  #Dopamine concentraation

        #self.I_AMPA = bm.Variable(bm.zeros(size))
        #self.I_NMDA = bm.Variable(bm.zeros(size))


    def I_SOMA(self):
        I_soma = (self.I_AMPA+self.I_NMDA)*(1+self.lambda_1)
        return I_soma

    def Q_Cl(self):
        Q_cl = bm.Variable(bm.zeros(self.size))
        return Q_cl

class D2Striatum(HumphriesNeuron):
    def __init__(self,
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=bp.init.ZeroInit(),
        noise=None,
        noise_type='value',
        keep_size=False,
        method='exp_auto',
        name=None,
        I_spon=0,
        V_lim=-20,
        lambda_2=0
    ):
        super().__init__(
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=V_initializer,
        noise=noise,
        noise_type=noise_type,
        keep_size=keep_size,
        method=method,
        name=name,
        I_spon=I_spon,
        V_lim=V_lim
        )

        self.lambda_2 = lambda_2  #Dopamine concentraation

        #self.I_AMPA = bm.Variable(bm.zeros(size))
        #self.I_NMDA = bm.Variable(bm.zeros(size))


    def I_SOMA(self):
        I_soma = (self.I_AMPA+self.I_NMDA)*(1-self.lambda_2)
        return I_soma

    def Q_Cl(self):
        Q_cl = bm.Variable(bm.zeros(self.size))
        return Q_cl


class STN(HumphriesNeuron):
    def __init__(self,
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=bp.init.ZeroInit(),
        noise=None,
        noise_type='value',
        keep_size=False,
        method='exp_auto',
        name=None,
        I_spon=0,
        V_lim=-20,
        lambda_2=0,
        #proportion=bm.Variable([0.33,0.33,0.34]),
        JP=1,
        JS=1,
        alpha=bm.Variable([0.5,0.5]),
        V_Ca=-10,
        J_Ca=90,
        t1=200,
        t2=1000
    ):
        super().__init__(
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=V_initializer,
        noise=noise,
        noise_type=noise_type,
        keep_size=keep_size,
        method=method,
        name=name,
        I_spon=I_spon,
        V_lim=V_lim
        )

        self.lambda_2 = lambda_2  #Dopamine concentration

        #self.I_AMPA = bm.Variable(bm.zeros(size))
        #self.I_NMDA = bm.Variable(bm.zeros(size))
        self.I_GABA_D = bm.Variable(bm.zeros(size))  #distal
        self.I_GABA_P = bm.Variable(bm.zeros(size))  #Proximal
        self.I_GABA_S = bm.Variable(bm.zeros(size))  #Somatic
        #self.proportion = proportion  #[D,P,S]
        self.JP = JP  #shunting inhibition parameter
        self.JS = JS  #shuting inhibition parameter
        self.alpha = alpha  #dopamine parameter

        #Ca current
        self.V_Ca = V_Ca
        self.J_Ca = J_Ca
        self.t1 = t1
        self.t2 = t2
        self.tb = bm.Variable(bm.ones(size) * -1e7)  #超过V_Ca的时刻

        self.hP = bm.Variable(bm.zeros(size))
        self.hS = bm.Variable(bm.zeros(size))

    def update_h(self):
        self.hP = (1-bm.abs(self.I_GABA_P)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_P)/self.JP,0)
        self.hS = (1-bm.abs(self.I_GABA_S)/self.JS)*bm.heaviside(1-bm.abs(self.I_GABA_S)/self.JS,0)

    def I_SOMA(self):
        #hP = (1-bm.abs(self.I_GABA_P)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_P)/self.JP)
        #hS = (1-bm.abs(self.I_GABA_S)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_S)/self.JP)

        I_soma = self.hS*self.hP*(self.I_AMPA+self.I_NMDA+self.I_GABA_D)

        return I_soma


    def Q_Cl(self):
        Q_cl = 1 - (self.hS+self.hP)/2
        return Q_cl

    def update(self,t,dt):
        refractory = (t - self.t_last_spike) <= self.tau_ref

        #dopamine
        self.I_AMPA = self.I_AMPA*(1-self.alpha[0]*self.lambda_2)
        self.I_NMDA = self.I_NMDA*(1-self.alpha[0]*self.lambda_2)
        self.I_GABA_D = self.I_GABA_D*(1-self.alpha[1]*self.lambda_2)
        self.I_GABA_P = self.I_GABA_P*(1-self.alpha[1]*self.lambda_2)
        self.I_GABA_S = self.I_GABA_S*(1-self.alpha[1]*self.lambda_2)

        #I_Ca
        delta_t = t-self.tb
        ramp_up = delta_t < self.t1
        ramp_down = delta_t < self.t1+self.t2
        I_up = bm.Variable(self.J_Ca*bm.ones(self.size))
        I_down = -self.J_Ca*(delta_t-self.t1)/self.t2 + self.J_Ca
        I_zero = bm.Variable(bm.zeros(self.size))
        I_Ca = bm.where(ramp_up,I_up,I_down)
        I_star = bm.where(ramp_down,I_Ca,I_zero)
        self.I_star = I_star

        #update V
        self.update_h()
        self.I_Cl = self.V_lim/self.R - self.I_spon
        V = self.integral(self.V, t, self.I_SOMA()+self.Q_Cl()*self.I_Cl+self.I_spon+self.I_star, dt=dt)

        #spike
        V = bm.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.spike.value = spike
        self.input[:] = 0.
        self.I_AMPA[:] = 0.
        self.I_NMDA[:] = 0.
        self.I_GABA_D[:] = 0.
        self.I_GABA_P[:] = 0.
        self.I_GABA_S[:] = 0.

        #Ca
        tb = self.V <= self.V_Ca
        self.tb.value = bm.where(tb,t,self.tb)
    

class GP(HumphriesNeuron):
    def __init__(self,
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=bp.init.ZeroInit(),
        noise=None,
        noise_type='value',
        keep_size=False,
        method='exp_auto',
        name=None,
        I_spon=0,
        V_lim=-20,
        lambda_2=0,
        #proportion=bm.Variable([0.33,0.33,0.34]),
        JP=1,
        JS=1,
        alpha=bm.Variable([0.5,0.5])
    ):
        super().__init__(
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=V_initializer,
        noise=noise,
        noise_type=noise_type,
        keep_size=keep_size,
        method=method,
        name=name,
        I_spon=I_spon,
        V_lim=V_lim
        )

        self.lambda_2 = lambda_2  #Dopamine concentration

        #self.I_AMPA = bm.Variable(bm.zeros(size))
        #self.I_NMDA = bm.Variable(bm.zeros(size))
        self.I_GABA_D = bm.Variable(bm.zeros(size))  #distal
        self.I_GABA_P = bm.Variable(bm.zeros(size))  #Proximal
        self.I_GABA_S = bm.Variable(bm.zeros(size))  #Somatic
        #self.proportion = proportion  #[D,P,S]
        self.JP = JP  #shunting inhibition parameter
        self.JS = JS  #shuting inhibition parameter
        self.alpha = alpha  #dopamine parameter


        self.hP = bm.Variable(bm.zeros(size))
        self.hS = bm.Variable(bm.zeros(size))

    def update_h(self):
        self.hP = (1-bm.abs(self.I_GABA_P)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_P)/self.JP,0)
        self.hS = (1-bm.abs(self.I_GABA_S)/self.JS)*bm.heaviside(1-bm.abs(self.I_GABA_S)/self.JS,0)

    def I_SOMA(self):
        #hP = (1-bm.abs(self.I_GABA_P)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_P)/self.JP)
        #hS = (1-bm.abs(self.I_GABA_S)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_S)/self.JP)

        I_soma = self.hS*self.hP*(self.I_AMPA+self.I_NMDA+self.I_GABA_D)

        return I_soma


    def Q_Cl(self):
        Q_cl = 1 - (self.hS+self.hP)/2
        return Q_cl

    def update(self,t,dt):
        refractory = (t - self.t_last_spike) <= self.tau_ref

        #dopamine
        self.I_AMPA = self.I_AMPA*(1-self.alpha[0]*self.lambda_2)
        self.I_NMDA = self.I_NMDA*(1-self.alpha[0]*self.lambda_2)
        self.I_GABA_D = self.I_GABA_D*(1-self.alpha[1]*self.lambda_2)
        self.I_GABA_P = self.I_GABA_P*(1-self.alpha[1]*self.lambda_2)
        self.I_GABA_S = self.I_GABA_S*(1-self.alpha[1]*self.lambda_2)

        #update V
        self.update_h()
        self.I_Cl = self.V_lim/self.R - self.I_spon
        V = self.integral(self.V, t, self.I_SOMA()+self.Q_Cl()*self.I_Cl+self.I_spon+self.I_star, dt=dt)

        #spike
        V = bm.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.spike.value = spike
        self.input[:] = 0.
        self.I_AMPA[:] = 0.
        self.I_NMDA[:] = 0.
        self.I_GABA_D[:] = 0.
        self.I_GABA_P[:] = 0.
        self.I_GABA_S[:] = 0.

class SNr(HumphriesNeuron):
    def __init__(self,
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=bp.init.ZeroInit(),
        noise=None,
        noise_type='value',
        keep_size=False,
        method='exp_auto',
        name=None,
        I_spon=0,
        V_lim=-20,
        #lambda_2=0,
        #proportion=bm.Variable([0.33,0.33,0.34]),
        JP=1,
        JS=1,
        #alpha=bm.Variable([0.5,0.5]),
    ):
        super().__init__(
        size,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer=V_initializer,
        noise=noise,
        noise_type=noise_type,
        keep_size=keep_size,
        method=method,
        name=name,
        I_spon=I_spon,
        V_lim=V_lim
        )

        #self.lambda_2 = lambda_2  #Dopamine concentration

        #self.I_AMPA = bm.Variable(bm.zeros(size))
        #self.I_NMDA = bm.Variable(bm.zeros(size))
        self.I_GABA_D = bm.Variable(bm.zeros(size))  #distal
        self.I_GABA_P = bm.Variable(bm.zeros(size))  #Proximal
        self.I_GABA_S = bm.Variable(bm.zeros(size))  #Somatic
        #self.proportion = proportion  #[D,P,S]
        self.JP = JP  #shunting inhibition parameter
        self.JS = JS  #shuting inhibition parameter
        #self.alpha = alpha  #dopamine parameter


        self.hP = bm.Variable(bm.zeros(size))
        self.hS = bm.Variable(bm.zeros(size))

    def update_h(self):
        self.hP = (1-bm.abs(self.I_GABA_P)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_P)/self.JP,0)
        self.hS = (1-bm.abs(self.I_GABA_S)/self.JS)*bm.heaviside(1-bm.abs(self.I_GABA_S)/self.JS,0)

    def I_SOMA(self):
        #hP = (1-bm.abs(self.I_GABA_P)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_P)/self.JP)
        #hS = (1-bm.abs(self.I_GABA_S)/self.JP)*bm.heaviside(1-bm.abs(self.I_GABA_S)/self.JP)

        I_soma = self.hS*self.hP*(self.I_AMPA+self.I_NMDA+self.I_GABA_D)

        return I_soma


    def Q_Cl(self):
        Q_cl = 1 - (self.hS+self.hP)/2
        return Q_cl

    def update(self,t,dt):
        refractory = (t - self.t_last_spike) <= self.tau_ref

        #dopamine
        '''self.I_AMPA = self.I_AMPA*(1-self.alpha[0]*self.lambda_2)
        self.I_NMDA = self.I_NMDA*(1-self.alpha[0]*self.lambda_2)
        self.I_GABA_D = self.I_GABA_D*(1-self.alpha[1]*self.lambda_2)
        self.I_GABA_P = self.I_GABA_P*(1-self.alpha[1]*self.lambda_2)
        self.I_GABA_S = self.I_GABA_S*(1-self.alpha[1]*self.lambda_2)'''

        #update V
        self.update_h()
        self.I_Cl = self.V_lim/self.R - self.I_spon
        V = self.integral(self.V, t, self.I_SOMA()+self.Q_Cl()*self.I_Cl+self.I_spon+self.I_star, dt=dt)

        #spike
        V = bm.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.spike.value = spike
        self.input[:] = 0.
        self.I_AMPA[:] = 0.
        self.I_NMDA[:] = 0.
        self.I_GABA_D[:] = 0.
        self.I_GABA_P[:] = 0.
        self.I_GABA_S[:] = 0.


class CortexNeuron(bp.dyn.PoissonGroup):
    def __init__(self, size: Shape, freqs: Union[float, jnp.ndarray, bm.JaxArray, Initializer], seed: int = None, keep_size: bool = False, name: str = None):
        super().__init__(size, freqs, seed, keep_size, name)
        self.modulator = bm.Variable(bm.ones(size))

    def update(self, t, dt):
        self.spike.update((self.rng.random(self.var_shape) <= (self.freqs * dt / 1000.))*self.modulator.astype(bm.bool_))

