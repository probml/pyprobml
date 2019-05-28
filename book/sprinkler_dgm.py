# water sprinkler DGM

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianModel([('C', 'S'), ('C', 'R'), ('S', 'W'), ('R', 'W')])

# Defining individual CPDs.
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.5, 0.5]])

# In pgmpy the columns are the evidences and rows are the states of the variable.
 
cpd_s = TabularCPD(variable='S', variable_card=2, 
                   values=[[0.5, 0.9],
                           [0.5, 0.1]],
                  evidence=['C'],
                  evidence_card=[2])

cpd_r = TabularCPD(variable='R', variable_card=2, 
                   values=[[0.8, 0.2],
                           [0.2, 0.8]],
                  evidence=['C'],
                  evidence_card=[2])

cpd_w = TabularCPD(variable='W', variable_card=2, 
                   values=[[1.0, 0.1, 0.1, 0.01],
                           [0.0, 0.9, 0.9, 0.99]],
                  evidence=['S', 'R'],
                  evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_c, cpd_s, cpd_r, cpd_w)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
model.check_model()

from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# p(R=1)= 0.5*0.2 + 0.5*0.8 = 0.5
print(infer.query(['R']) ['R'])

# P(R=1|W=1) = 0.7079
print(infer.query(['R'], evidence={'W': 1}) ['R'])


# P(R=1|W=1,S=1) = 0.3204
print(infer.query(['R'], evidence={'W': 1, 'S': 1}) ['R'])



