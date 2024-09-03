import numpy as np
import pandas as pd
from bit_config import bit_config
from calculate import calculate_bops_mobilenet, calculate_size_mobilenet, calculate_bops_resnet50, calculate_size_resnet50

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


caculate_dict = {'mobilenetv2_w1': {'bops': calculate_bops_mobilenet, 'size': calculate_size_mobilenet},
                 'resnet50': {'bops': calculate_bops_resnet50, 'size': calculate_size_resnet50}}


class SizeProblem(Problem):
    def __init__(self, num_variable, arch, limit, df, layer_num_df):
        super().__init__(n_var=num_variable // 2,
                         n_obj=2,
                         n_ieq_constr=1, 
                         n_eq_constr=0,
                         xl=np.array([4] * (num_variable // 2)),
                         xu=np.array([8] * (num_variable // 2)),
                         vtype=int)
        self.num_variable = num_variable
        self.arch = arch
        self.limit = limit
        self.df = df
        self.layer_num_df = layer_num_df

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = np.sum(self.df['DeltaAcc'].values[1::2] * (8 - X) / 4, axis=1)
        f2 = np.sum(self.df['DeltaAcc'].values[1::2] * (8 - X) / 4, axis=1) + np.sum(self.df['Influence_fillna'].values[1::2] * (8 - X) / 4, axis=1) + [self.layer_num_df.loc[self.layer_num_df['Layer_num'] == layer_num, 'Avg_corr'].values[0] for layer_num in np.sum(X != 8, axis=1)]

        g1 = [caculate_dict[self.arch]['size'](np.zeros((self.num_variable // 2)) + 8, X[i, :]) - self.limit for i in range(X.shape[0])]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1])


class BOPsProblem(Problem):
    def __init__(self, num_variable, arch, limit_value, df, layer_num_df):
        super().__init__(n_var=num_variable,
                         n_obj=2,
                         n_ieq_constr=1, 
                         n_eq_constr=0,
                         xl=np.array([4] * num_variable),
                         xu=np.array([8] * num_variable),
                         vtype=int)
        self.num_variable = num_variable
        self.arch = arch
        self.limit = limit_value
        self.df = df
        self.layer_num_df = layer_num_df

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = np.sum(self.df['DeltaAcc'].values * (8 - X) / 4, axis=1)
        f2 = np.sum(self.df['DeltaAcc'].values * (8 - X) / 4, axis=1) + np.sum(self.df['Influence_fillna'].values * (8 - X) / 4, axis=1) + [self.layer_num_df.loc[self.layer_num_df['Layer_num'] == layer_num, 'Avg_corr'].values[0] for layer_num in np.sum(X != 8, axis=1)]

        g1 = [caculate_dict[self.arch]['bops'](X[i, ::2], X[i, 1::2]) - self.limit for i in range(X.shape[0])]

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1])


def nsga(arch, limit, limit_p, df, layer_num_df):
    layers = df['Layer'].values
    num_variable = len(layers)

    nsga_dict = {'Config_name': [],
                 'Config': []}

    config_name = limit + str(limit_p)
    
    limit_8bit = caculate_dict[arch][limit](np.zeros((num_variable//2)) + 8, np.zeros((num_variable//2)) + 8)
    limit_4bit = caculate_dict[arch][limit](np.zeros((num_variable//2)) + 4, np.zeros((num_variable//2)) + 4)
    limit_value = limit_4bit + limit_p * (limit_8bit - limit_4bit)

    if limit == "size":
        problem = SizeProblem(num_variable, arch, limit_value, df, layer_num_df)
    else:
        problem = BOPsProblem(num_variable, arch, limit_value, df, layer_num_df)

    algorithm = NSGA2(pop_size=100,
                      sampling=IntegerRandomSampling(),
                      crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      eliminate_duplicates=True)
    result = minimize(problem, 
                      algorithm, 
                      termination=('n_gen', 100), 
                      seed=1,
                      verbose=False)

    if result.X is None:
        if limit == "size":
            print(f'Model size should be limited to between {int(limit_4bit)} and {int(limit_8bit)}')
        else:
            print(f'BOPs should be limited to between {int(limit_4bit)} and {int(limit_8bit)}')
        return None
    
    for i in range(len(result.X)):
        nsga_dict['Config_name'].append(config_name + '_' + str(i + 1))
        nsga_dict['Config'].append(result.X[i])

    nsga_df = pd.DataFrame(nsga_dict)

    model_config = bit_config[arch]
    changed_layers = {}
    
    for _, row in nsga_df.iterrows():
        if limit == "size":
            config = [8 if index % 2 == 0 else row['Config'][index//2] for index in range(num_variable)]
        else:
            config = row['Config']
        changed_layers['bit_config_' + arch + '_' + row['Config_name']] = {}
        for index, bit in enumerate(config):
            if bit != 8:
                changed_layers['bit_config_' + arch + '_' + row['Config_name']][layers[index]] = bit
    output_bit_config = {}

    for model, paras in changed_layers.items():
        output_bit_config[model] = model_config.copy()
        for para in paras:
            output_bit_config[model][para] = paras[para]
    
    return output_bit_config