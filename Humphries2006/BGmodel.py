import brainpy as bp
import brainpy.math as bm
from brainpy.dyn import channels, synapses, synouts, synplast

    

class HumphriesNeuron(bp.dyn.neurons.LIF):
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
        V_lim=-20
    ):
        super().__init__(size,
        V_rest,
        V_rest,
        V_th,
        R,
        tau,
        tau_ref,
        V_initializer,
        noise,
        keep_size,
        method,
        name)

        self.V_lim = V_lim
        #currents to be specified in specific neuron types
        self.I_spon = bp.init.init_param(I_spon,size)
        self.I_star = bm.Variable(bm.zeros(size))
        self.I_Cl = bm.Variable(bm.zeros(size))

        self.I_AMPA = bm.Variable(bm.zeros(size))
        self.I_NMDA = bm.Variable(bm.zeros(size))

        '''#currents from synapses
        self.I_Distal = bm.Variable(bm.zeros(size))
        self.I_Proximal = bm.Variable(bm.zeros(size))
        self.I_Soma = bm.Variable(bm.zeros(size))'''
        
    #shunting inhibition: return hS*hP*ID
    def I_SOMA(self):
        pass

    #return the coeffient of the I_Cl
    def Q_Cl(self):
        pass

    def update(self,t,dt):
        refractory = (t - self.t_last_spike) <= self.tau_ref
        self.I_Cl = self.V_lim/self.R - self.I_spon
        V = self.integral(self.V, t, self.I_SOMA()+self.Q_Cl()*self.I_Cl+self.I_spon+self.I_star, dt=dt)
        V = bm.where(refractory, self.V, V)
        spike = V >= self.V_th
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.spike.value = spike
        self.input[:] = 0.
        self.I_AMPA[:] = 0.
        self.I_NMDA[:] = 0.
