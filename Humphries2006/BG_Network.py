import sys
import brainpy as bp
import brainpy.math as bm
import brainpy.initialize as bi
import sys

sys.path.append('./')
import BG_neurons as BGn
import BG_synapse as BGs

bm.set_platform('cpu')

neuron_num = 64
channel_num = 3
V_rest = 0
tau_ref = 2
V_lim = -20
rho = 0.25
dt = 0.1
eta = 0.5


#Striatum
V_th_Striatum = 30
R_init_Striatum = bi.Normal(42,4.2)
tau_init_Striatum = bi.Normal(25,2.5)
I_spon_Striatum = -0.25
lambda_1 = 0.3
lambda_2 = 0.3

#STN
V_th_STN = 20
R_init_STN = bi.Normal(18,1.8)
tau_init_STN = bi.Normal(6,0.6)
alpha_STN = [0.5,0.25]
V_Ca_init_STN = bi.Normal(-10,1)
J_Ca_init_STN = bi.Normal(90,9)
t1_init_STN = bi.Normal(200,20)
t2_init_STN = bi.Normal(1000,100)
I_spon_STN = 1.1

#GP
V_th_GP = 30
R_init_GP = bi.Normal(88,8.8)
tau_init_GP = bi.Normal(14,1.4)
beta_GP = [0.5,0.5]
I_spon_GP = 0.38

#SNr
V_th_SNr = 30
R_init_SNr = bi.Normal(112,11.2)
tau_init_SNr = bi.Normal(8,0.8)
I_spon_SNr = 0.39

tau_AMPA = 2
tau_GABA = 3
tau_NMDA = 100

#D1-SNr GABA
gmax_DS = 3/28
delay_DS = int(4/dt)#.astype(bm.int_)
pro_DS = [0,0,1]

#D2-GP GABA
gmax_DGP = 3/22
delay_DGP = int(5/dt)#.astype(bm.int_)
pro_DGP = [0.33,0.33,0.34]

#GP-STN GABA
gmax_GPSTN = 1/6
delay_GPSTN = int(4/dt)#.astype(bm.int_)
pro_GPSTN = [0.3,0.4,0.3]

#GP-GP GABA
gmax_GG = 3/88
delay_GG = int(1/dt)#.astype(bm.int_)
pro_GG = [0.5,0.5,0]

#GP-SNr GABA
gmax_GSNr = 3/112
delay_GSNr = int(3/dt)#.astype(bm.int_)
pro_GSNr = [0.5,0.5,0]

#SNr-SNr GABA
gmax_SS = 3/112
delay_SS = int(1/dt)#.astype(bm.int_)
pro_SS = [0.5,0.5,0]

#STN-SNr
gmax_STNSNr_AMPA = 3/112
gmax_STNSNr_NMDA = 0.1/112
delay_STNSNr = int(1.5/dt)#.astype(bm.int_)

#STN-GP
gmax_STNGP_AMPA = 3/88
gmax_STNGP_NMDA = 0.1/88
delay_STNGP = int(2/dt)#.astype(bm.int_)


#cortex D
f1 = 3
f2 = 3
f3 = 3
gmax_CD_AMPA = 1/14
delay_CD = int(10/dt)
gmax_CD_NMDA = 0.1/42

#cortex STN
gmax_CS_AMPA = 1/6
gmax_CS_NMDA = 0.1/18
delay_CS = int(2.5/dt)


class BasalGanglia(bp.dyn.Network):
    def __init__(self, *ds_tuple, name=None, **ds_dict):
        super().__init__(*ds_tuple, name=name, **ds_dict)

        self.Cortex_1 = BGn.CortexNeuron(neuron_num,f1)
        self.Cortex_2 = BGn.CortexNeuron(neuron_num,f2)
        self.Cortex_3 = BGn.CortexNeuron(neuron_num,f3)

        self.D1Striatum_1 = BGn.D1Striatum(neuron_num,V_rest,V_th_Striatum,R_init_Striatum([neuron_num],bm.float64),tau_init_Striatum([neuron_num],bm.float64),tau_ref,I_spon=I_spon_Striatum,V_lim=V_lim,lambda_1=lambda_1)
        self.D1Striatum_2 = BGn.D1Striatum(neuron_num,V_rest,V_th_Striatum,R_init_Striatum([neuron_num],bm.float64),tau_init_Striatum([neuron_num],bm.float64),tau_ref,I_spon=I_spon_Striatum,V_lim=V_lim,lambda_1=lambda_1)
        self.D1Striatum_3 = BGn.D1Striatum(neuron_num,V_rest,V_th_Striatum,R_init_Striatum([neuron_num],bm.float64),tau_init_Striatum([neuron_num],bm.float64),tau_ref,I_spon=I_spon_Striatum,V_lim=V_lim,lambda_1=lambda_1)

        self.D2Striatum_1 = BGn.D2Striatum(neuron_num,V_rest,V_th_Striatum,R_init_Striatum([neuron_num],bm.float64),tau_init_Striatum([neuron_num],bm.float64),tau_ref,I_spon=I_spon_Striatum,V_lim=V_lim,lambda_2=lambda_2)
        self.D2Striatum_2 = BGn.D2Striatum(neuron_num,V_rest,V_th_Striatum,R_init_Striatum([neuron_num],bm.float64),tau_init_Striatum([neuron_num],bm.float64),tau_ref,I_spon=I_spon_Striatum,V_lim=V_lim,lambda_2=lambda_2)
        self.D2Striatum_3 = BGn.D2Striatum(neuron_num,V_rest,V_th_Striatum,R_init_Striatum([neuron_num],bm.float64),tau_init_Striatum([neuron_num],bm.float64),tau_ref,I_spon=I_spon_Striatum,V_lim=V_lim,lambda_2=lambda_2)

        self.STN_1 = BGn.STN(neuron_num,V_rest,V_th_STN,R_init_STN([neuron_num]),tau_init_STN([neuron_num]),tau_ref,I_spon=I_spon_STN,lambda_2=lambda_2,alpha=bm.Variable(alpha_STN),V_Ca=V_Ca_init_STN([neuron_num]),J_Ca=J_Ca_init_STN([neuron_num]),t1=t1_init_STN([neuron_num]),t2=t2_init_STN([neuron_num]))
        self.STN_2 = BGn.STN(neuron_num,V_rest,V_th_STN,R_init_STN([neuron_num]),tau_init_STN([neuron_num]),tau_ref,I_spon=I_spon_STN,lambda_2=lambda_2,alpha=bm.Variable(alpha_STN),V_Ca=V_Ca_init_STN([neuron_num]),J_Ca=J_Ca_init_STN([neuron_num]),t1=t1_init_STN([neuron_num]),t2=t2_init_STN([neuron_num]))
        self.STN_3 = BGn.STN(neuron_num,V_rest,V_th_STN,R_init_STN([neuron_num]),tau_init_STN([neuron_num]),tau_ref,I_spon=I_spon_STN,lambda_2=lambda_2,alpha=bm.Variable(alpha_STN),V_Ca=V_Ca_init_STN([neuron_num]),J_Ca=J_Ca_init_STN([neuron_num]),t1=t1_init_STN([neuron_num]),t2=t2_init_STN([neuron_num]))

        self.GP_1 = BGn.GP(neuron_num,V_rest,V_th_GP,R_init_GP([neuron_num]),tau_init_GP([neuron_num]),tau_ref,I_spon=I_spon_GP,lambda_2=lambda_2,alpha=bm.Variable(beta_GP))
        self.GP_2 = BGn.GP(neuron_num,V_rest,V_th_GP,R_init_GP([neuron_num]),tau_init_GP([neuron_num]),tau_ref,I_spon=I_spon_GP,lambda_2=lambda_2,alpha=bm.Variable(beta_GP))
        self.GP_3 = BGn.GP(neuron_num,V_rest,V_th_GP,R_init_GP([neuron_num]),tau_init_GP([neuron_num]),tau_ref,I_spon=I_spon_GP,lambda_2=lambda_2,alpha=bm.Variable(beta_GP))

        self.SNr_1 = BGn.SNr(neuron_num,V_rest,V_th_SNr,R_init_SNr([neuron_num]),tau_init_SNr([neuron_num]),tau_ref,I_spon=I_spon_SNr)
        self.SNr_2 = BGn.SNr(neuron_num,V_rest,V_th_SNr,R_init_SNr([neuron_num]),tau_init_SNr([neuron_num]),tau_ref,I_spon=I_spon_SNr)
        self.SNr_3 = BGn.SNr(neuron_num,V_rest,V_th_SNr,R_init_SNr([neuron_num]),tau_init_SNr([neuron_num]),tau_ref,I_spon=I_spon_SNr)

        self.C_D1_AMPA_1 = BGs.BG_AMPA(self.Cortex_1,self.D1Striatum_1,bp.conn.FixedProb(rho),g_max=gmax_CD_AMPA,delay_step=delay_CD,tau=tau_AMPA)
        self.C_D1_AMPA_2 = BGs.BG_AMPA(self.Cortex_2,self.D1Striatum_2,bp.conn.FixedProb(rho),g_max=gmax_CD_AMPA,delay_step=delay_CD,tau=tau_AMPA)
        self.C_D1_AMPA_3 = BGs.BG_AMPA(self.Cortex_3,self.D1Striatum_3,bp.conn.FixedProb(rho),g_max=gmax_CD_AMPA,delay_step=delay_CD,tau=tau_AMPA)

        self.C_D1_NMDA_1 = BGs.BG_NMDA(self.Cortex_1,self.D1Striatum_1,bp.conn.FixedProb(rho),g_max=gmax_CD_NMDA,delay_step=delay_CD,tau=tau_NMDA)
        self.C_D1_NMDA_2 = BGs.BG_NMDA(self.Cortex_2,self.D1Striatum_2,bp.conn.FixedProb(rho),g_max=gmax_CD_NMDA,delay_step=delay_CD,tau=tau_NMDA)
        self.C_D1_NMDA_3 = BGs.BG_NMDA(self.Cortex_3,self.D1Striatum_3,bp.conn.FixedProb(rho),g_max=gmax_CD_NMDA,delay_step=delay_CD,tau=tau_NMDA)

        self.C_D2_AMPA_1 = BGs.BG_AMPA(self.Cortex_1,self.D2Striatum_1,bp.conn.FixedProb(rho),g_max=gmax_CD_AMPA,delay_step=delay_CD,tau=tau_AMPA)
        self.C_D2_AMPA_2 = BGs.BG_AMPA(self.Cortex_2,self.D2Striatum_2,bp.conn.FixedProb(rho),g_max=gmax_CD_AMPA,delay_step=delay_CD,tau=tau_AMPA)
        self.C_D2_AMPA_3 = BGs.BG_AMPA(self.Cortex_3,self.D2Striatum_3,bp.conn.FixedProb(rho),g_max=gmax_CD_AMPA,delay_step=delay_CD,tau=tau_AMPA)

        self.C_D2_NMDA_1 = BGs.BG_NMDA(self.Cortex_1,self.D2Striatum_1,bp.conn.FixedProb(rho),g_max=gmax_CD_NMDA,delay_step=delay_CD,tau=tau_NMDA)
        self.C_D2_NMDA_2 = BGs.BG_NMDA(self.Cortex_2,self.D2Striatum_2,bp.conn.FixedProb(rho),g_max=gmax_CD_NMDA,delay_step=delay_CD,tau=tau_NMDA)
        self.C_D2_NMDA_3 = BGs.BG_NMDA(self.Cortex_3,self.D2Striatum_3,bp.conn.FixedProb(rho),g_max=gmax_CD_NMDA,delay_step=delay_CD,tau=tau_NMDA)

        self.C_STN_AMPA_1 = BGs.BG_AMPA(self.Cortex_1,self.STN_1,bp.conn.FixedProb(rho),g_max=gmax_CS_AMPA,delay_step=delay_CS,tau=tau_AMPA)
        self.C_STN_AMPA_2 = BGs.BG_AMPA(self.Cortex_2,self.STN_2,bp.conn.FixedProb(rho),g_max=gmax_CS_AMPA,delay_step=delay_CS,tau=tau_AMPA)
        self.C_STN_AMPA_3 = BGs.BG_AMPA(self.Cortex_3,self.STN_3,bp.conn.FixedProb(rho),g_max=gmax_CS_AMPA,delay_step=delay_CS,tau=tau_AMPA)

        self.C_STN_NMDA_1 = BGs.BG_NMDA(self.Cortex_1,self.STN_1,bp.conn.FixedProb(rho),g_max=gmax_CS_NMDA,delay_step=delay_CS,tau=tau_NMDA)
        self.C_STN_NMDA_2 = BGs.BG_NMDA(self.Cortex_2,self.STN_2,bp.conn.FixedProb(rho),g_max=gmax_CS_NMDA,delay_step=delay_CS,tau=tau_NMDA)
        self.C_STN_NMDA_3 = BGs.BG_NMDA(self.Cortex_3,self.STN_3,bp.conn.FixedProb(rho),g_max=gmax_CS_NMDA,delay_step=delay_CS,tau=tau_NMDA)

        self.D1_SNr_1 = BGs.BG_GABA(self.D1Striatum_1,self.SNr_1,bp.conn.FixedProb(rho),g_max=gmax_DS,delay_step=delay_DS,tau=tau_GABA,proportion=bm.Variable(pro_DS))
        self.D1_SNr_2 = BGs.BG_GABA(self.D1Striatum_2,self.SNr_2,bp.conn.FixedProb(rho),g_max=gmax_DS,delay_step=delay_DS,tau=tau_GABA,proportion=bm.Variable(pro_DS))
        self.D1_SNr_3 = BGs.BG_GABA(self.D1Striatum_3,self.SNr_3,bp.conn.FixedProb(rho),g_max=gmax_DS,delay_step=delay_DS,tau=tau_GABA,proportion=bm.Variable(pro_DS))

        self.D2_GP_1 = BGs.BG_GABA(self.D2Striatum_1,self.GP_1,bp.conn.FixedProb(rho),g_max=gmax_DGP,delay_step=delay_DGP,tau=tau_GABA,proportion=bm.Variable(pro_DGP))
        self.D2_GP_2 = BGs.BG_GABA(self.D2Striatum_2,self.GP_2,bp.conn.FixedProb(rho),g_max=gmax_DGP,delay_step=delay_DGP,tau=tau_GABA,proportion=bm.Variable(pro_DGP))
        self.D2_GP_3 = BGs.BG_GABA(self.D2Striatum_3,self.GP_3,bp.conn.FixedProb(rho),g_max=gmax_DGP,delay_step=delay_DGP,tau=tau_GABA,proportion=bm.Variable(pro_DGP))

        self.GP_STN_1 = BGs.BG_GABA(self.GP_1,self.STN_1,bp.conn.FixedProb(rho),g_max=gmax_GPSTN,delay_step=delay_GPSTN,tau=tau_GABA,proportion=bm.Variable(pro_GPSTN))
        self.GP_STN_2 = BGs.BG_GABA(self.GP_2,self.STN_2,bp.conn.FixedProb(rho),g_max=gmax_GPSTN,delay_step=delay_GPSTN,tau=tau_GABA,proportion=bm.Variable(pro_GPSTN))
        self.GP_STN_3 = BGs.BG_GABA(self.GP_3,self.STN_3,bp.conn.FixedProb(rho),g_max=gmax_GPSTN,delay_step=delay_GPSTN,tau=tau_GABA,proportion=bm.Variable(pro_GPSTN))

        self.GP_GP_11 = BGs.BG_GABA(self.GP_1,self.GP_1,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_12 = BGs.BG_GABA(self.GP_1,self.GP_2,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_13 = BGs.BG_GABA(self.GP_1,self.GP_3,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_21 = BGs.BG_GABA(self.GP_2,self.GP_1,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_22 = BGs.BG_GABA(self.GP_2,self.GP_2,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_23 = BGs.BG_GABA(self.GP_2,self.GP_3,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_31 = BGs.BG_GABA(self.GP_3,self.GP_1,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_32 = BGs.BG_GABA(self.GP_3,self.GP_2,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))
        self.GP_GP_33 = BGs.BG_GABA(self.GP_3,self.GP_3,bp.conn.FixedProb(rho),g_max=gmax_GG,delay_step=delay_GG,tau=tau_GABA,proportion=bm.Variable(pro_GG))

        self.GP_SNr_1 = BGs.BG_GABA(self.GP_1,self.SNr_1,bp.conn.FixedProb(rho),g_max=gmax_GSNr,delay_step=delay_GSNr,tau=tau_GABA,proportion=bm.Variable(pro_GSNr))
        self.GP_SNr_2 = BGs.BG_GABA(self.GP_2,self.SNr_2,bp.conn.FixedProb(rho),g_max=gmax_GSNr,delay_step=delay_GSNr,tau=tau_GABA,proportion=bm.Variable(pro_GSNr))
        self.GP_SNr_3 = BGs.BG_GABA(self.GP_3,self.SNr_3,bp.conn.FixedProb(rho),g_max=gmax_GSNr,delay_step=delay_GSNr,tau=tau_GABA,proportion=bm.Variable(pro_GSNr))

        self.SNr_SNr_11 = BGs.BG_GABA(self.SNr_1,self.SNr_1,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_12 = BGs.BG_GABA(self.SNr_1,self.SNr_2,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_13 = BGs.BG_GABA(self.SNr_1,self.SNr_3,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_21 = BGs.BG_GABA(self.SNr_2,self.SNr_1,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_22 = BGs.BG_GABA(self.SNr_2,self.SNr_2,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_23 = BGs.BG_GABA(self.SNr_2,self.SNr_3,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_31 = BGs.BG_GABA(self.SNr_3,self.SNr_1,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_32 = BGs.BG_GABA(self.SNr_3,self.SNr_2,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))
        self.SNr_SNr_33 = BGs.BG_GABA(self.SNr_3,self.SNr_3,bp.conn.FixedProb(rho),g_max=gmax_SS,delay_step=delay_SS,tau=tau_GABA,proportion=bm.Variable(pro_SS))

        self.STN_SNr_AMPA_11 = BGs.BG_AMPA(self.STN_1,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_12 = BGs.BG_AMPA(self.STN_1,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_13 = BGs.BG_AMPA(self.STN_1,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_21 = BGs.BG_AMPA(self.STN_2,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_22 = BGs.BG_AMPA(self.STN_2,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_23 = BGs.BG_AMPA(self.STN_2,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_31 = BGs.BG_AMPA(self.STN_3,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_32 = BGs.BG_AMPA(self.STN_3,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)
        self.STN_SNr_AMPA_33 = BGs.BG_AMPA(self.STN_3,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_AMPA,delay_step=delay_STNSNr,tau=tau_AMPA)

        self.STN_SNr_NMDA_11 = BGs.BG_NMDA(self.STN_1,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_12 = BGs.BG_NMDA(self.STN_1,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_13 = BGs.BG_NMDA(self.STN_1,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_21 = BGs.BG_NMDA(self.STN_2,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_22 = BGs.BG_NMDA(self.STN_2,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_23 = BGs.BG_NMDA(self.STN_2,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_31 = BGs.BG_NMDA(self.STN_3,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_32 = BGs.BG_NMDA(self.STN_3,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)
        self.STN_SNr_NMDA_33 = BGs.BG_NMDA(self.STN_3,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNSNr_NMDA,delay_step=delay_STNSNr,tau=tau_NMDA)

        self.STN_GP_AMPA_11 = BGs.BG_AMPA(self.STN_1,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_12 = BGs.BG_AMPA(self.STN_1,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_13 = BGs.BG_AMPA(self.STN_1,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_21 = BGs.BG_AMPA(self.STN_2,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_22 = BGs.BG_AMPA(self.STN_2,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_23 = BGs.BG_AMPA(self.STN_2,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_31 = BGs.BG_AMPA(self.STN_3,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_32 = BGs.BG_AMPA(self.STN_3,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)
        self.STN_GP_AMPA_33 = BGs.BG_AMPA(self.STN_3,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_AMPA,delay_step=delay_STNGP,tau=tau_AMPA)

        self.STN_GP_NMDA_11 = BGs.BG_NMDA(self.STN_1,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_12 = BGs.BG_NMDA(self.STN_1,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_13 = BGs.BG_NMDA(self.STN_1,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_21 = BGs.BG_NMDA(self.STN_2,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_22 = BGs.BG_NMDA(self.STN_2,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_23 = BGs.BG_NMDA(self.STN_2,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_31 = BGs.BG_NMDA(self.STN_3,self.SNr_1,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_32 = BGs.BG_NMDA(self.STN_3,self.SNr_2,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)
        self.STN_GP_NMDA_33 = BGs.BG_NMDA(self.STN_3,self.SNr_3,bp.conn.FixedProb(rho/channel_num),g_max=gmax_STNGP_NMDA,delay_step=delay_STNGP,tau=tau_NMDA)

        self.update_J()

    def update_J(self):
        nodes = self.nodes(level=1, include_self=False)
        nodes = nodes.subset(bp.dyn.DynamicalSystem)
        nodes = nodes.unique()
        synapse_groups = nodes.subset(BGs.BG_GABA)

        conn_s_dict = {}
        conn_p_dict = {}
        neugroup_dict = {}

        for gaba in synapse_groups.values():
            name = gaba.post.name
            conn = gaba.conn.require('conn_mat').astype(bm.int_).sum(axis=1)
            g = gaba.g_max
            mask_s = gaba.mask_S.astype(bm.int_)
            mask_p = gaba.mask_P.astype(bm.int_)
            conn_s = conn*mask_s*g
            conn_p = conn*mask_p*g

            if name in neugroup_dict.keys():
                conn_s_dict[name] += conn_s
                conn_p_dict[name] += conn_p
            else:
                conn_s_dict[name] = conn_s
                conn_p_dict[name] = conn_p
                neugroup_dict[name] = gaba.post

        for name in neugroup_dict.keys():
            JS = eta*bm.median(conn_s_dict[name])
            JP = eta*bm.median(conn_p_dict[name])

            neugroup_dict[name].JS = JS if JS>0 else 1e7
            neugroup_dict[name].JP = JP if JP>0 else 1e7

            '''JS = eta*bm.median(conn*mask_s)*g
            JP = eta*bm.median(conn*mask_p)*g
            gaba.post.JS = JS
            gaba.post.JP = JP'''


if __name__ == '__main__':

    T = 5000

    T_win = 500

    current1 = bp.inputs.section_input(values=[0, 1.],
                                             durations=[1000, 4000],
                                             dt=dt)

    current2= bp.inputs.section_input(values=[0, 1.],
                                             durations=[2500, 2500],
                                             dt=dt)

#,inputs=[('Cortex_1.modulator',current1,'iter','='),('Cortex_2.modulator',current2,'iter','=')]
    BG_Network = BasalGanglia()
    runner = bp.dyn.DSRunner(BG_Network,monitors=['D1_SNr_2.g','Cortex_1.spike','D2Striatum_1.spike','D2Striatum_2.spike','D2Striatum_3.spike','STN_1.spike','STN_2.spike','STN_3.spike','GP_1.spike','GP_2.spike','GP_3.spike','SNr_1.spike','SNr_2.spike','SNr_3.spike','Cortex_2.spike','Cortex_3.spike']) #,dt=dt)
    runner.run(T)
    import matplotlib.pyplot as plt

    plt.figure()
    bp.visualize.line_plot(runner.mon.ts,runner.mon['D1_SNr_2.g'][:,24])
    '''
    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['SNr_1.spike'],title='SNr1', show=False)

    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['SNr_2.spike'],title='SNr2', show=False)

    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['SNr_3.spike'],title='SNr3', show=False)


    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['D2Striatum_1.spike'],title='D1S', show=False)


    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['STN_1.spike'],title='STN', show=False)


    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['GP_1.spike'],title='GP', show=False)

    plt.figure()
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['Cortex_1.spike'],title='Cortex', show=True)'''

    SNr_1_sp = runner.mon['SNr_1.spike']

    SNr_2_sp = runner.mon['SNr_2.spike']

    SNr_3_sp = runner.mon['SNr_3.spike']

    SNr_1_fr = bp.measure.firing_rate(SNr_1_sp,T_win,dt=dt)

    SNr_2_fr = bp.measure.firing_rate(SNr_2_sp,T_win,dt=dt)

    SNr_3_fr = bp.measure.firing_rate(SNr_3_sp,T_win,dt=dt)

    t = bm.arange(0,SNr_1_fr.shape[0],1)

    plt.figure()
    bp.visualize.line_plot(t,SNr_1_fr, show=False,legend='SNr 1')
    bp.visualize.line_plot(t,SNr_2_fr, show=False,legend='SNr 2')
    bp.visualize.line_plot(t,SNr_3_fr, show=False,legend='SNr 3')
    plt.legend()

    
    Cortex_1_sp = runner.mon['Cortex_1.spike']
    Cortex_2_sp = runner.mon['Cortex_2.spike']
    Cortex_3_sp = runner.mon['Cortex_3.spike']

    Cortex_1_fr = bp.measure.firing_rate(Cortex_1_sp,T_win,dt=dt)

    Cortex_2_fr = bp.measure.firing_rate(Cortex_2_sp,T_win,dt=dt)

    Cortex_3_fr = bp.measure.firing_rate(Cortex_3_sp,T_win,dt=dt)

    plt.figure()
    bp.visualize.line_plot(t,Cortex_1_fr, show=False,legend='Cortex 1')
    bp.visualize.line_plot(t,Cortex_2_fr, show=False,legend='Cortex 2')
    bp.visualize.line_plot(t,Cortex_3_fr, show=False,legend='Cortex 3')
    plt.legend()

    D2Striatum_1_sp = runner.mon['D2Striatum_1.spike']
    D2Striatum_2_sp = runner.mon['D2Striatum_2.spike']
    D2Striatum_3_sp = runner.mon['D2Striatum_3.spike']

    D2Striatum_1_fr = bp.measure.firing_rate(D2Striatum_1_sp,T_win,dt=dt)

    D2Striatum_2_fr = bp.measure.firing_rate(D2Striatum_2_sp,T_win,dt=dt)

    D2Striatum_3_fr = bp.measure.firing_rate(D2Striatum_3_sp,T_win,dt=dt)

    plt.figure()
    bp.visualize.line_plot(t,D2Striatum_1_fr, show=False,legend='D2Striatum 1')
    bp.visualize.line_plot(t,D2Striatum_2_fr, show=False,legend='D2Striatum 2')
    bp.visualize.line_plot(t,D2Striatum_3_fr, show=False,legend='D2Striatum 3')
    plt.legend()
    
    STN_1_sp = runner.mon['STN_1.spike']
    STN_2_sp = runner.mon['STN_2.spike']
    STN_3_sp = runner.mon['STN_3.spike']

    STN_1_fr = bp.measure.firing_rate(STN_1_sp,T_win,dt=dt)

    STN_2_fr = bp.measure.firing_rate(STN_2_sp,T_win,dt=dt)

    STN_3_fr = bp.measure.firing_rate(STN_3_sp,T_win,dt=dt)

    plt.figure()
    bp.visualize.line_plot(t,STN_1_fr, show=False,legend='STN 1')
    bp.visualize.line_plot(t,STN_2_fr, show=False,legend='STN 2')
    bp.visualize.line_plot(t,STN_3_fr, show=False,legend='STN 3')
    plt.legend()

    GP_1_sp = runner.mon['GP_1.spike']
    GP_2_sp = runner.mon['GP_2.spike']
    GP_3_sp = runner.mon['GP_3.spike']

    GP_1_fr = bp.measure.firing_rate(GP_1_sp,T_win,dt=dt)

    GP_2_fr = bp.measure.firing_rate(GP_2_sp,T_win,dt=dt)

    GP_3_fr = bp.measure.firing_rate(GP_3_sp,T_win,dt=dt)

    plt.figure()
    bp.visualize.line_plot(t,GP_1_fr, show=False,legend='GP 1')
    bp.visualize.line_plot(t,GP_2_fr, show=False,legend='GP 2')
    bp.visualize.line_plot(t,GP_3_fr, show=True,legend='GP 3')
    plt.legend()





